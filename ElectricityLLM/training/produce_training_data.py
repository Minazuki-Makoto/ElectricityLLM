import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Literal

from openai import OpenAI
from pydantic import BaseModel, Field, model_validator


# 修正：按脚本位置计算绝对路径，原来的 "...\\LLM-data" 不是有效项目路径。
PROJECT_DIR = Path(__file__).resolve().parents[2]
QUERY_PATH = PROJECT_DIR / "LLM-data" / "training_data" / "chunk-questions-1000.txt"
OUTPUT_PATH = PROJECT_DIR / "LLM-data" / "training_data" / "output-data-1000.json"

# 修正：脚本直接运行时，确保能够导入项目内的 ElasticSearch 包。
CODE_DIR = PROJECT_DIR / "ElectricityLLM"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from ElasticSearch.Search.test import full_find


# 这是“数据生成器”的提示词，只用于调用 GPT 生成训练样本，不写进最终 SFT messages。
GENERATOR_PROMPT = """
你是电力系统领域的高质量 SFT 数据生成器。用户消息中包含若干带 source 标签的检索资料和一个问题。
你的任务是仅依据这些资料生成专业、准确、完整且自然的训练答案，并返回规定的结构化字段。

一、资料边界与事实约束
1. 只能使用输入中明确提供的事实、参数、公式、标准名称、因果关系和工程要求。
2. 禁止使用模型自身知识补充资料中没有的内容；即使补充内容在常识上可能正确，也不得写入答案。
3. 禁止编造、猜测或修正标准编号、数值、设备类型、适用范围、故障原因、试验结论和引用来源。
4. 检索结果可能包含跨主题、重复或仅有关键词重合的噪声。只使用与当前问题直接相关的资料，
   不得为了增加篇幅拼接无关的设备选型、运行维护或通用电力知识。
5. 多个 source 之间存在冲突时，不得自行裁决或融合为确定结论；应说明现有资料存在不一致，
   并据此判断资料是否足以回答。

二、问题覆盖与答案完整性
6. 生成答案前，必须在内部列出问题中的全部显式子任务，并逐项确认资料能否支持，但不要输出分析过程。
   只要资料能够支持，就必须在 answer 中为每个子任务提供可辨认的对应内容，不得合并后遗漏。
7. 以下复合问法采用硬性覆盖要求：
   - “核心机理、关键影响因素及工程注意事项”：三项必须分别回答；
   - “主要特性、可能风险及工程要求”：三项必须分别回答，风险不得被注意事项替代；
   - “原理和注意事项”：必须同时说明工作或作用机理，以及运行、检查或安全注意事项；
   - “关键结论及逻辑关系”：先提炼结论，再明确说明因果、条件、并列、递进或约束关系；
   - “工作原理与应用边界”：必须同时说明如何工作、适用于什么条件，以及不适用或受限条件；
   - “设备选型、运行或故障判断”：资料支持的每个指定应用维度都要分别给出可执行要点；
   - 比较类问题：逐一覆盖全部比较对象，说明相同点、差异点及适用条件，不得只介绍其中一个对象。
8. 不得只回答问题的一部分，也不得用一句笼统总结代替资料已经支持的关键内容。
   每个核心子任务至少应给出一个由资料直接支持的具体结论；资料包含条件、风险、例外或限制时一并保留。
9. answer 长度由有效资料量和 difficulty 共同决定，并以中文字符数作为生成时的近似参考：
   - easy：通常 150～350 个中文字符；
   - medium：通常 350～650 个中文字符；
   - hard：通常 550～1000 个中文字符。
   复合问题优先保证逐项覆盖，可在资料确实丰富时适度超过上限；不得为满足长度重复题目、堆砌同义句、
   引入无关知识，也不得为了简短而遗漏关键条件、限制、风险、步骤或逻辑关系。
10. 对定义类问题，说明对象的含义、主要作用或组成；对原理类问题，说明过程、关系和关键影响因素；
    对比较类问题，明确比较对象、主要差异和适用条件；对步骤、维护和故障类问题，按资料说明前提、
    操作或检查重点、风险及注意事项；对计算类问题，只使用资料给出的公式、变量和条件。
11. 如果资料提供多个结论，应明确说明它们之间的逻辑关系，不要只把句子机械罗列在一起。
    涉及标准体系时，应区分基础原则、产品要求、试验要求、运行维护要求及其适用层级，不得只罗列标准名称。
12. answer 必须是完整结束的最终文本，最后一个有效字符应为“。！？”等结束标点；
    不得在半句话、冒号、未完成的列表项、未闭合括号或 Markdown 标记处结束。

三、资料充分性
13. 只有当检索资料能够支持问题的全部核心要求时，content_sufficient 才能为 true。
14. content_sufficient 为 true 时：
    - answer 必须完整覆盖资料能够支持的全部核心要求；
    - used_sources 必须至少包含一个真实且实际用于答案的 source 标签。
15. 如果缺少回答任一核心要求所必需的事实，content_sufficient 必须为 false：
    - answer 应清楚指出哪些核心内容缺少资料支持；若资料仍能支持部分内容，可简要说明能够确认的部分，
      但不得把部分回答描述为完整答案；
    - 不得使用模型自身知识补全答案；
    - 如果 answer 使用了资料中仍可确认的部分，used_sources 必须列出实际支撑这些内容的 source；
    - 只有当资料完全无法支持 answer 中的任何事实、answer 仅说明缺少信息时，used_sources 才为空。
16. 不能仅因资料没有使用与问题完全相同的标题或措辞就判定不足；只要资料在语义上明确支持答案即可。
    同样，也不能仅因关键词相同就判定资料充分。

四、引用约束
17. used_sources 只能选择输入中真实存在且确实支撑 answer 的 source 标签，例如 source_1。
18. 不得改写、拼接或编造 source 标签；不得选择只与问题有表面关键词重合但未用于答案的 source。
19. 不要在 answer 中自行输出 source 标签、document_title、chunk_id、检索分数或引用清单；
    程序会根据 used_sources 映射真实来源。

五、表达要求
20. answer 应直接进入技术内容，不使用“根据检索资料”“根据上述内容”“本段主要介绍”等元话语。
21. 语言应严谨、清晰、自然。复杂问题应使用与问题子任务对应的简短编号或小标题，保证覆盖关系可检查；
    但避免不必要的多级标题、
    分隔线、Markdown 表格、emoji、重复结论和装饰性格式。
22. 每个句子应语义完整，枚举项之间关系清楚，专业缩写、数值和单位保持与资料一致。
23. 不得输出与问题无关的建议，不得在资料不足后推荐资料中未出现的标准、文献或产品。
24. answer 字段中只写最终答案内容，不写推理过程、覆盖检查过程或自我评价。

六、标签选择
25. question_type 必须从 definition、principle、reasoning、comparison、procedure、maintenance、
    fault_analysis、calculation、other 中选择最匹配的一项。
26. difficulty 根据问题包含的子任务数量和完成答案所需的资料整合程度判断：
    - easy：单一事实、定义或直接作用；
    - medium：需要整合多个相关事实、条件或因果关系，或包含两个明确子任务；
    - hard：需要跨多个有效 source 比较、推理、计算，或同时覆盖三个及以上工程维度。

七、输出前内部质量检查
27. 返回结构化字段前必须在内部完成以下检查，但不得把检查过程写入 answer：
    - 问题中的每个显式子任务是否都有对应回答；
    - 每项结论是否能由至少一个 selected source 直接支持；
    - 是否遗漏风险、边界、条件、逻辑关系或用户指定的应用维度；
    - answer 长度是否与 difficulty 和有效资料量匹配；
    - 是否存在重复、无关扩写或资料之外的事实；
    - 最后一句是否语义完整并以结束标点结句。
""".strip()

# 这是最终用于微调的 system prompt，不包含“输出 metadata JSON”等数据生产指令。
SFT_SYSTEM_PROMPT = (
    "你是严谨的电力系统领域知识助手。只能依据用户提供的检索资料回答，不得使用资料之外的知识补充、"
    "猜测或编造事实、数值、标准和结论。回答前应识别问题中的全部显式子任务；资料能够支持时，必须逐项"
    "覆盖，不得遗漏原理、特性、影响因素、风险、工程要求、应用边界或逻辑关系等用户明确要求的维度。"
    "复杂问题使用简短编号或小标题组织，每个子任务给出具体且有资料依据的结论，并保留必要的条件、限制"
    "和例外。答案长度应与问题复杂度和有效资料量匹配，不得过度压缩，也不得重复凑字。资料不足时，应明确"
    "指出无法支持的具体内容，不得用模型知识补全。回答应专业、准确、清晰、语义完整，并以结束标点结句。"
)

MODEL_NAME = os.getenv("SFT_DATA_MODEL", "gpt-5.4-mini")
# 扩大重排前的召回候选池，提高自然用户问法命中正确 chunk 的概率。
# 最终仍只把 CONTEXT_TOPK 条写入 SFT 样本，避免无关资料和上下文截断。
RETRIEVAL_TOPK = 40
CONTEXT_TOPK = 5
MAX_RETRIES = 3


class TrainingSample(BaseModel):
    answer: str = Field(min_length=10)
    # 修正：模型只选择短 source 标签，禁止模型手写并改坏索引中的标题或 chunk_id。
    used_sources: list[str] = Field(
        default_factory=list,
        description=(
            "实际支撑 answer 中事实的 source 标签；部分资料可用时仍应列出，"
            "完全没有使用任何资料时才为空"
        ),
    )
    content_sufficient: bool = Field(
        description="检索资料是否足以覆盖问题的全部核心要求"
    )
    question_type: Literal[
        "definition",
        "principle",
        "reasoning",
        "comparison",
        "procedure",
        "maintenance",
        "fault_analysis",
        "calculation",
        "other",
    ]
    difficulty: Literal["easy", "medium", "hard"]

    @model_validator(mode="after")
    def validate_response(self):
        self.answer = self.answer.strip()
        self.used_sources = list(dict.fromkeys(x.strip() for x in self.used_sources if x.strip()))

        # 完全充足的答案必须有引用；部分充足时允许保留支撑已确认内容的引用。
        if self.content_sufficient and not self.used_sources:
            raise ValueError("资料充足时 used_sources 不能为空")
        return self


def create_client() -> OpenAI:
    # 修正：os.environ[...] 缺失时会直接 KeyError；这里给出明确错误信息。
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 OPENAI_API_KEY 环境变量")
    return OpenAI(api_key=api_key)


def read_queries(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"query 文件不存在：{path}")

    # 修正：queries.txt 是一行一个 query 的纯文本，不能使用 json.load()。
    queries = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not queries:
        raise ValueError("query 文件中没有有效问题")

    # 去重并保持原有顺序，避免重复调用 API。
    return list(dict.fromkeys(queries))


def compact_chunks(chunks: list[Dict[str, Any]]) -> list[Dict[str, str]]:
    compacted = []
    for chunk in chunks[:CONTEXT_TOPK]:
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        text = str(chunk.get("text", "")).strip()
        if not chunk_id or not text:
            continue
        compacted.append(
            {
                "document_title": str(chunk.get("title", "")).strip(),
                "chunk_id": chunk_id,
                "text": text,
            }
        )
    return compacted


def build_generation_input(query: str, chunks: list[Dict[str, str]]) -> list[Dict[str, str]]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 不能为空")

    sources = []
    for index, chunk in enumerate(chunks, start=1):
        sources.append(
            f"[source_{index}]\n"
            f"document_title: {chunk['document_title']}\n"
            f"chunk_id: {chunk['chunk_id']}\n"
            f"text: {chunk['text']}"
        )
    context = "\n\n".join(sources) if sources else "（没有检索到可用资料）"

    # 修正：chunks 属于当前用户提供的上下文，不应使用 assistant 角色伪装成模型历史回复。
    user_content = f"【检索资料】\n{context}\n\n【问题】\n{query.strip()}"
    return [
        {"role": "system", "content": GENERATOR_PROMPT},
        {"role": "user", "content": user_content},
    ]


def get_answer(client: OpenAI, messages: list[Dict[str, str]]) -> TrainingSample:
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # 修正：使用 Pydantic 结构化输出，避免用 json_repair 把错误响应“修”成貌似合法的数据。
            response = client.responses.parse(
                model=MODEL_NAME,
                input=messages,
                text_format=TrainingSample,
            )
            if response.output_parsed is None:
                raise ValueError("模型没有返回可解析的结构化结果")
            return response.output_parsed
        except Exception as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"调用模型连续失败 {MAX_RETRIES} 次") from last_error


def resolve_citations(
    result: TrainingSample,
    chunks: list[Dict[str, str]],
) -> tuple[list[str], list[str]]:
    """将 source 标签映射为索引原值，保证标题和 chunk_id 逐字符一致。"""
    by_source = {f"source_{index}": chunk for index, chunk in enumerate(chunks, start=1)}
    unknown_sources = [source for source in result.used_sources if source not in by_source]
    if unknown_sources:
        raise ValueError(f"模型引用了未提供的 source：{unknown_sources}")

    cited_chunks = [by_source[source] for source in result.used_sources]
    document_titles = list(
        dict.fromkeys(chunk["document_title"] for chunk in cited_chunks)
    )
    chunk_ids = [chunk["chunk_id"] for chunk in cited_chunks]
    return document_titles, chunk_ids


def build_sft_user_message(query: str, chunks: list[Dict[str, str]]) -> str:
    sources = []
    for index, chunk in enumerate(chunks, start=1):
        sources.append(
            f"[source_{index}]\n"
            f"document_title: {chunk['document_title']}\n"
            f"chunk_id: {chunk['chunk_id']}\n"
            f"{chunk['text']}"
        )
    return f"【检索资料】\n{'\n\n'.join(sources)}\n\n【问题】\n{query}"


def dataset_split(index: int) -> str:
    # 每连续 10 条按 8:1:1 分配，避免按文件顺序切分导致验证集主题单一。
    remainder = index % 10
    if remainder == 9:
        return "validation"
    if remainder == 0:
        return "test"
    return "train"


def generate_training_sample(
    client: OpenAI,
    elasticsearch_uris: str,
    elasticsearch_username: str,
    elasticsearch_password: str,
    elastic_index: str,
    query: str,
    idx: int,
) -> Dict[str, Any] | None:
    retrieved = full_find(
        elasticsearch_uris=elasticsearch_uris,
        elasticsearch_username=elasticsearch_username,
        elasticsearch_password=elasticsearch_password,
        elastic_index=elastic_index,
        query=query,
        document_title_weight=0.10,
        title_weight=0.20,
        text_weight=0.02,
        vector_weight=0.60,
        description_text_weight=0.08,
        topk=RETRIEVAL_TOPK,
        score_threshold=0.30,
    )
    chunks = compact_chunks(retrieved)

    # 基于现有知识库生成的 query 理应能够召回资料。若 Top-K 过滤后为空，
    # 说明 query 质量或检索存在问题：直接跳过，不调用 GPT，也不写入 SFT。
    if not chunks:
        return None

    # source 标签不合法时也重试，避免一次偶发引用错误让整个批处理退出。
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = get_answer(client, build_generation_input(query, chunks))
            document_titles, chunk_ids = resolve_citations(result, chunks)
            break
        except ValueError as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                raise RuntimeError("模型连续返回无效引用") from last_error
            time.sleep(2 ** (attempt - 1))

    # 修正：sample_id 原来把 str 与 int 直接相加；使用固定六位编号与参考格式一致。
    return {
        "messages": [
            {"role": "system", "content": SFT_SYSTEM_PROMPT},
            {"role": "user", "content": build_sft_user_message(query, chunks)},
            {"role": "assistant", "content": result.answer},
        ],
        "metadata": {
            "sample_id": f"sft_{idx:06d}",
            # 修正：字段名与参考格式统一为复数；删除尾随逗号，避免值变成单元素 tuple。
            # 这里的值直接来自 Elasticsearch 返回结果，不采用模型生成的字符串。
            "document_titles": document_titles,
            "chunk_ids": chunk_ids,
            "context_sufficiency": result.content_sufficient,
            "question_type": result.question_type,
            "difficulty": result.difficulty,
            "dataset_split": dataset_split(idx),
        },
    }


def load_existing_samples(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"已有输出不是 JSON 数组：{path}")
    return data


def save_samples(path: Path, samples: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> None:
    queries = read_queries(QUERY_PATH)
    client = create_client()
    samples = load_existing_samples(OUTPUT_PATH)

    # 修正：支持断点续跑，并在每条成功后原子写盘，避免中途失败丢失全部 API 结果。
    completed_ids = {
        sample.get("metadata", {}).get("sample_id")
        for sample in samples
        if isinstance(sample, dict)
    }
    for index, query in enumerate(queries, start=91):
        sample_id = f"sft_{index:06d}"
        if sample_id in completed_ids:
            print(f"[{index}/{len(queries)}] skip {sample_id}")
            continue

        sample = generate_training_sample(
            client=client,
            elasticsearch_uris="https://localhost:9200",
            elasticsearch_username=os.environ["ELASTICSEARCH_USERNAME"],
            elasticsearch_password=os.environ["ELASTICSEARCH_PASSWORD"],
            elastic_index="electricity-infos",
            query=query,
            idx=index,
        )

        # 空检索样本不进入训练集，也不会产生 GPT API 费用。
        if sample is None:
            print(f"[{index}/{len(queries)}] skipped {sample_id}: no retrieved chunks")
            continue

        samples.append(sample)
        save_samples(OUTPUT_PATH, samples)
        print(f"[{index}/{len(queries)}] saved {sample_id}")


if __name__ == "__main__":
    main()
