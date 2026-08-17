import os
import json
import logging
from time import perf_counter
from threading import Lock
from flask import Flask,request,jsonify

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

from Agent.Graph import agentGraph
from redis_stream_save import (
    end_put,
    publish_delta,
    publish_error,
    start_put,
)


neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD")
MODEL_NAME = os.getenv("GLM_MODEL_NAME", "glm-5.2")
SMALL_MODEL_NAME = os.getenv("GLM_SMALL_MODEL_NAME", "glm-4.7-flashx")
IMPORTANT_SMALL_MODEL_NAME = os.getenv(
    "GLM_IMPORTANT_SMALL_MODEL_NAME",
    "glm-4.5-air",
)

ELASTICSEARCH_URIS = os.getenv("ELASTICSEARCH_URIS", "https://localhost:9200")
ELASTICSEARCH_USERNAME = os.environ["ELASTICSEARCH_USERNAME"]
ELASTICSEARCH_PASSWORD = os.environ["ELASTICSEARCH_PASSWORD"]
elastic_index = os.getenv("ELASTICSEARCH_TEXT_INDEX", "electricity-infos")
elastic_plot_index = os.getenv("ELASTICSEARCH_PLOT_INDEX", "electricity-plot-data")
light_graph_index = os.getenv("ELASTICSEARCH_GRAPH_INDEX", "graph_index")

logger.info("正在初始化 ElectricityLLM 模型与检索组件")
agent = agentGraph(ELASTICSEARCH_URIS,
                   ELASTICSEARCH_USERNAME,
                   ELASTICSEARCH_PASSWORD,
                   elastic_index,
                   light_graph_index,
                   elastic_plot_index,
                   neo4j_uri,
                   neo4j_username,
                   neo4j_password,
                   model_name=MODEL_NAME,
                   small_model_name=SMALL_MODEL_NAME,
                   important_small_model_name=IMPORTANT_SMALL_MODEL_NAME,
                   )
agent_lock = Lock()
logger.info("ElectricityLLM 初始化完成")

app = Flask(__name__)
app.json.ensure_ascii = False


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "UP"}), 200


@app.route("/memory/index", methods=["POST"])
def index_completed_chat_memory():
    """仅为 MySQL 已提交为 COMPLETED 的聊天建立 ES 记忆索引。"""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"status": "failed", "message": "请求体不是json格式"}), 400

    try:
        chat_id = int(data.get("chat_id"))
        user_id = int(data.get("user_id"))
        session_id = int(data.get("session_id"))
    except (TypeError, ValueError):
        return jsonify({"status": "failed", "message": "chat_id、user_id和session_id必须是整数"}), 400

    if chat_id <= 0 or user_id <= 0 or session_id <= 0:
        return jsonify({"status": "failed", "message": "chat_id、user_id和session_id必须大于0"}), 400

    with agent_lock:
        result = agent.write_in_memory(
            {
                "external_chat_id": chat_id,
                "user_id": user_id,
                "session_id": session_id,
            }
        )
    if result.get("indexed") is True:
        return jsonify({"status": "success", "message": result}), 200
    if result.get("reason") == "chat_not_completed":
        return jsonify({"status": "skipped", "message": result}), 409
    return jsonify({"status": "failed", "message": result}), 500

@app.route("/ai", methods=[ 'POST'])
def chat_with_ai():
    started_at = perf_counter()
    request_task_id = None
    sequence = 0
    stream_metadata = {}
    stream_started = False
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify(
                {
                    "task_id":None,
                    "status":"failed",
                    "message":"请求体不是json格式"
                }
            ) ,400

        query = data.get("query")
        user_id = data.get("user_id")
        session_id = data.get("session_id")
        chat_id = data.get("chat_id")
        request_task_id = str(data.get("task_id", "")).strip()

        stream_metadata = {
            "user_id": str(user_id),
            "session_id": str(session_id),
        }

        logger.info(
            "收到 /ai 请求 task_id=%s user_id=%s session_id=%s query_length=%d",
            request_task_id,
            user_id,
            session_id,
            len(query) if isinstance(query, str) else 0,
        )

        if not isinstance(query, str) or not query.strip():
            return jsonify(
                {
                    "status":"failed",
                    "message":"请求体中未包含用户的问题"
                }
            ),400

        if not user_id:
            return jsonify(
                {
                    "status":"failed",
                    "message":"请求体中未包含用户的id"
                }
            ),400

        if not session_id:
            return jsonify(
                {
                    "status":"failed",
                    "message":"请求体中未包含会话的id"
                }
            ),400

        if not request_task_id:
            return jsonify(
                {
                    "task_id":None,
                    "status":"failed",
                    "message":"请求体中未包含任务id"
                }
            ),400

        if chat_id is not None:
            try:
                chat_id = int(chat_id)
            except (TypeError, ValueError) as error:
                raise ValueError("chat_id 必须是整数") from error
            if chat_id <= 0:
                raise ValueError("chat_id 必须大于 0")

        streamed_answer_parts = []

        def handle_delta(content: str) -> None:
            nonlocal sequence

            if not isinstance(content, str) or not content:
                return

            sequence += 1
            streamed_answer_parts.append(content)
            publish_delta(
                task_id=request_task_id,
                sequence=sequence,
                content=content,
                metadata=stream_metadata,
            )

        logger.info("任务等待模型锁 task_id=%s", request_task_id)
        with agent_lock:

            logger.info("任务获得模型锁 task_id=%s", request_task_id)
            start_put(
                task_id=request_task_id,
                content="",
                metadata=stream_metadata,
            )

            stream_started = True
            answer = agent.invoke(
                query=query,
                user_id=user_id,
                session_id=session_id,
                task_id=request_task_id,
                chat_id=chat_id,
                on_delta=handle_delta,
            )

        if not isinstance(answer, dict):
            raise RuntimeError("Agent 返回结果不是字典")

        final_results = answer.get("final_results", [])
        if not isinstance(final_results, list):
            raise RuntimeError("Agent final_results 不是列表")

        ordered_results = sorted(
            final_results,
            key=lambda item: (
                item.get("task_id", 0)
                if isinstance(item, dict)
                else 0
            )
        )
        response = []
        combined_answers = []

        for item in ordered_results:
            if not isinstance(item, dict):
                continue

            sub_task_id = item.get("task_id")
            sub_question = item.get("sub_question", {})
            if not isinstance(sub_question, dict):
                sub_question = {}

            question = str(
                sub_question.get("question", "")
            ).strip()
            skill = str(
                sub_question.get("skill", "chat")
            ).strip().lower()
            raw_answer = str(
                item.get("final_answer", "")
            ).strip()
            task_error = str(item.get("error", "") or "").strip()

            task_answer = raw_answer
            artifacts = []

            if skill == "plot":
                try:
                    plot_result = json.loads(raw_answer)
                    if not isinstance(plot_result, dict):
                        raise ValueError("plot 结果不是 JSON 对象")

                    answer_data = plot_result.get("answer", "")
                    task_answer = answer_data

                    artifacts = plot_result.get("artifacts", [])
                    if not isinstance(artifacts, list):
                        artifacts = []
                except (json.JSONDecodeError, TypeError, ValueError):
                    task_answer = raw_answer

            if not task_answer and task_error:
                task_answer = f"子任务执行失败：{task_error}"

            response.append(
                {
                    "task_id":sub_task_id,
                    "question":question,
                    "skill":skill,
                    "task_answer":task_answer,
                    "artifacts":artifacts,
                    "error":task_error or None
                }
            )

            if task_answer:
                if question:
                    combined_answers.append(
                        f"{sub_task_id}. {question}\n{task_answer}"
                    )
                else:
                    combined_answers.append(task_answer)

        final_ans = "\n\n".join(combined_answers)
        if not final_ans:
            final_ans = str(
                answer.get("final_answer", "")
            ).strip()

        if not final_ans:
            raise RuntimeError("Agent 没有生成最终回答")

        streamed_answer = "".join(streamed_answer_parts).strip()
        if streamed_answer and streamed_answer not in final_ans:
            logger.warning(
                "流式回答与最终回答不一致 task_id=%s "
                "streamed_length=%d final_length=%d",
                request_task_id,
                len(streamed_answer),
                len(final_ans),
            )

        if not streamed_answer:
            sequence += 1
            publish_delta(
                task_id=request_task_id,
                sequence=sequence,
                content=final_ans,
                metadata=stream_metadata,
            )

        sequence += 1
        end_put(
            task_id=request_task_id,
            content="",
            sequence=sequence,
            metadata=stream_metadata,
        )

        logger.info(
            "/ai 请求成功 task_id=%s user_id=%s session_id=%s details=%d elapsed=%.2fs",
            request_task_id,
            user_id,
            session_id,
            len(response),
            perf_counter() - started_at,
        )

        return jsonify(
            {
                "task_id":request_task_id,
                "status":"success",
                "message":{
                    "final_answer":final_ans,
                    "details":response
                }
            }
        ),200

    except ValueError as e:
        logger.warning(
            "/ai 请求参数或业务校验失败 elapsed=%.2fs error=%s",
            perf_counter() - started_at,
            e,
        )
        if stream_started and request_task_id:
            try:
                publish_error(
                    task_id=request_task_id,
                    sequence=sequence + 1,
                    error_code="AGENT_VALIDATION_FAILED",
                    error_message="AI 任务处理失败",
                    metadata=stream_metadata,
                )

            except Exception:
                logger.exception(
                    "写入 Redis error 事件失败 task_id=%s",
                    request_task_id,
                )

        return jsonify(
            {
                "task_id":request_task_id,
                "status":"failed",
                "message":str(e)
            }
        ) , 400

    except Exception as e:
        logger.exception(
            "/ai 请求异常 elapsed=%.2fs",
            perf_counter() - started_at,
        )
        if request_task_id:
            try:
                publish_error(
                    task_id=request_task_id,
                    sequence=sequence + 1,
                    error_code="AGENT_GENERATION_FAILED",
                    error_message="AI 回答生成失败",
                    metadata=stream_metadata,
                )
            except Exception:
                logger.exception(
                    "写入 Redis error 事件失败 task_id=%s",
                    request_task_id,
                )

        return jsonify(
            {
                "task_id":request_task_id,
                "status":"failed",
                "message":"服务器内部错误"
            }
        ) , 500


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
