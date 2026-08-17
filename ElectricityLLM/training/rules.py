import re
from collections.abc import Sequence
from difflib import SequenceMatcher
from FlagEmbedding import FlagReranker
from pathlib import Path
from typing import Any
import torch


ROOT_PATH = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_PATH / 'LLM-data'/"Models"/"reranker-model"

rerank_model = FlagReranker(
    model_name_or_path=str(MODEL_PATH),
    use_fp16=False
)


# 只针对不自然的“元话语”扣分，不把正确的资料不足说明列为禁用词。
META_PHRASES = (
    "根据检索资料",
    "根据提供的资料",
    "根据上述资料",
    "从检索结果来看",
    "作为一个人工智能",
    "作为 AI",
)

REFUSAL_PHRASES = (
    "无法确定",
    "无法判断",
    "资料不足",
    "信息不足",
    "无法从现有资料中确定",
    "无法从现有信息中确定",
)

def compute_related_score(
    query: str,
    answer: str,
    normalize: bool = True,
) -> float:
    """计算单个 query-answer 文本对的相关性分数。"""

    model = rerank_model.model
    tokenizer = rerank_model.tokenizer

    model.eval()
    device = next(model.parameters()).device

    inputs = tokenizer(
        query,
        text_pair=answer,
        return_tensors="pt",
        truncation=True,
        max_length=8192,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(
            **inputs,
            return_dict=True,
        )

        logits = outputs.logits

        if logits.shape != (1, 1):
            raise RuntimeError(
                "当前函数要求模型对单个文本对输出一个 logit，"
                f"实际形状为 {tuple(logits.shape)}"
            )

        score_tensor = logits.squeeze(-1).float()

        if normalize:
            score_tensor = torch.sigmoid(score_tensor)

        score = score_tensor.item()

    return score

def normalize_answer(answer: str) -> str:
    """统一处理奖励函数的输入，避免 None 或非字符串导致训练中断。"""
    if not isinstance(answer, str):
        return ""
    return answer.strip()


def normalize_for_similarity(text: str) -> str:
    """为问题复制检测标准化文本。

    忽略大小写、空白、标点和 Markdown 格式符号，保留中文、
    英文字母和数字，避免模型只添加句号就绕过检查。
    """
    if not isinstance(text, str):
        return ""

    return re.sub(
        r"[^a-z0-9\u4e00-\u9fff]+",
        "",
        text.lower().strip(),
    )


def query_copy_penalty(query: str, answer: str) -> float:
    """
    惩罚完全复制、轻微改写或只有少量扩写的问题复述。

    正常的专业回答即使包含问题中的关键词，只要增加了足够的
    有效内容，就不会被此规则惩罚。
    """
    normalized_query = normalize_for_similarity(query)
    normalized_answer = normalize_for_similarity(answer)

    if not normalized_query or not normalized_answer:
        return 0.0

    if normalized_query == normalized_answer:
        return -1.0

    # 只调整原问字符顺序，仍然没有增加新信息。
    if sorted(normalized_query) == sorted(normalized_answer):
        return -0.8

    query_length = len(normalized_query)
    answer_length = len(normalized_answer)
    length_ratio = answer_length / query_length
    similarity = SequenceMatcher(
        None,
        normalized_query,
        normalized_answer,
        autojunk=False,
    ).ratio()

    if length_ratio <= 1.2 and similarity >= 0.90:
        return -0.8

    if length_ratio <= 1.8 and similarity >= 0.80:
        return -0.4

    if normalized_query in normalized_answer:
        added_length = answer_length - query_length
        if added_length < 10:
            return -0.8
        if added_length < 20:
            return -0.4

    return 0.0


def _extract_context_text(context: str | dict[str, Any]) -> str:
    """同时支持纯文本资料和包含 text 字段的检索结果字典。"""
    if isinstance(context, str):
        return context.strip()
    if isinstance(context, dict):
        return str(context.get("text", "")).strip()
    return ""


def context_copy_penalty(context: str, answer: str) -> float:
    """惩罚完全照抄或几乎完全照抄单条检索资料的回答。"""
    normalized_context = normalize_for_similarity(context)
    normalized_answer = normalize_for_similarity(answer)

    if not normalized_context or not normalized_answer:
        return 0.0

    if normalized_context == normalized_answer:
        return -1.0

    length_ratio = len(normalized_answer) / len(normalized_context)
    if 0.8 <= length_ratio <= 1.2:
        similarity = SequenceMatcher(
            None,
            normalized_context,
            normalized_answer,
            autojunk=False,
        ).ratio()
        if similarity >= 0.95:
            return -0.8

    return 0.0


def context_grounding_reward(
    answer: str,
    retrieved_contexts: Sequence[str | dict[str, Any]],
    context_sufficient: bool,
) -> float:
    """
    奖励回答与预检索资料的相关性，同时防止整段照抄资料。

    每条资料分别与回答计分，最终取最高相关分，因为一个正确
    回答可能只需由其中一条资料充分支持。
    """
    text = normalize_answer(answer)
    if not text:
        return -1.0

    # 资料不足时由拒答规则评分，避免两个奖励互相抵消。
    if not context_sufficient:
        return 0.0

    context_texts = [
        context_text
        for context in retrieved_contexts
        if (context_text := _extract_context_text(context))
    ]
    if not context_texts:
        return 0.0

    copy_penalty = min(
        context_copy_penalty(context, text)
        for context in context_texts
    )
    if copy_penalty < 0.0:
        return copy_penalty

    return max(
        compute_related_score(context, text)
        for context in context_texts
    )


def basic_quality_reward(
    answer: str,
    min_length: int = 10,
    ideal_max_length: int = 1500,
    hard_max_length: int = 2500,
) -> float:
    """奖励非空、长度合理的回答，但不鼓励无限增加文本。"""
    text = normalize_answer(answer)
    if not text:
        return -1.0

    length = len(text)
    if length < min_length:
        return -0.6
    if length > hard_max_length:
        return -0.5
    if length > ideal_max_length:
        return 0.0
    return 0.3


def meta_phrase_penalty(answer: str) -> float:
    """惩罚影响自然度的元话语，惩罚上限为 -1.0。"""
    text = normalize_answer(answer)
    if not text:
        return 0.0

    hit_count = sum(phrase in text for phrase in META_PHRASES)
    return max(-1.0, -0.25 * hit_count)


def repetition_penalty(answer: str) -> float:
    """检测完全重复的句子，避免模型通过复读填充长度。"""
    text = normalize_answer(answer)
    if not text:
        return 0.0

    sentences = [
        sentence.strip()
        for sentence in re.split(r"[\u3002！？!?\n]+", text)
        if sentence.strip()
    ]
    if len(sentences) < 2:
        return 0.0

    duplicate_count = len(sentences) - len(set(sentences))
    return max(-1.0, -0.25 * duplicate_count)


def sentence_completion_reward(answer: str) -> float:
    """检查回答是否以完整句标点结束。

    允许句号后带有 Markdown 加粗等闭合符号，例如“结论。**”。
    这只能判断形式上的完整性，不能完全代替语义完整性判断。
    """
    text = normalize_answer(answer)
    if not text:
        return -1.0

    normalized_end = re.sub(
        r"[\s*_~`”’\"》〉）\)\]\}]+$",
        "",
        text,
    )

    if normalized_end.endswith(("。", ".", "！", "!", "？", "?")):
        return 0.3

    return -0.8


def refusal_consistency_reward(
    answer: str,
    context_sufficient: bool,
) -> float:
    """
    根据检索资料是否充分，奖励正确作答或正确拒答。

    context_sufficient 必须来自可靠的数据标签，不能根据模型回答反推。
    """
    text = normalize_answer(answer)
    if not text:
        return -1.0

    contains_refusal = any(phrase in text for phrase in REFUSAL_PHRASES)

    if context_sufficient:
        return -1.0 if contains_refusal else 0.3

    return 1.0 if contains_refusal else -1.0

def combined_rule_reward(
    answer: str,
    query: str,
    retrieved_contexts: Sequence[str | dict[str, Any]],
    context_sufficient: bool,
) -> float:

    """汇总所有规则奖励，并将结果限制在 [-2.0, 2.0] 内。"""
    total = (
        basic_quality_reward(answer)
        + meta_phrase_penalty(answer)
        + repetition_penalty(answer)
        + sentence_completion_reward(answer)
        + refusal_consistency_reward(answer, context_sufficient)
        + compute_related_score(query, answer)
        + query_copy_penalty(query, answer)
        + context_grounding_reward(
            answer,
            retrieved_contexts,
            context_sufficient,
        )
    )

    return max(-2.0, min(2.0, total))
