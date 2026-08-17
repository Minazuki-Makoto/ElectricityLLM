from __future__ import annotations

import json
from typing import Any
import os
import redis
from json_repair import repair_json
import pymysql
from pymysql import cursors


MAX_HISTORY = int(os.getenv("SHORT_MEMORY_MAX_HISTORY", "20"))
HISTORY_TTL_SECONDS = int(os.getenv("SHORT_MEMORY_TTL_SECONDS", str(24 * 60 * 60)))

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "127.0.0.1"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    db=int(os.getenv("REDIS_DB", "0")),
    password=os.getenv("REDIS_PASSWORD") or None,
    decode_responses=True,
)

DATABASE_NAME = os.getenv("MYSQL_DATABASE", "scms")

def generate_key(user_id: Any, session_id: Any) -> str:
    """根据 Java 端传入的可信用户和会话标识生成 Redis Key。"""
    """这里的key和java中的key保持一致"""
    user = str(user_id).strip()
    session = str(session_id).strip()
    if not user or not session:
        raise ValueError("user_id 和 session_id 不能为空")
    return f"chat:list:user:{user}:session:{session}"

def get_memory(
    user_id: Any,
    session_id: Any
) -> list:

    key = generate_key(user_id, session_id)
    history_list = redis_client.lrange(key, -MAX_HISTORY, -1)
    if not history_list:
        history_list = retrieve_history(user_id,session_id)
        
        if not history_list:
            return []

    new_history_list = []
    for history in history_list:
        try:
            neo_history = (
                history
                if isinstance(history, dict)
                else json.loads(history)
            )
        except json.JSONDecodeError:
            continue
        if not isinstance(neo_history, dict):
            continue

        if neo_history.get("status","FAILED") == "FAILED" or neo_history.get("status","FAILED") == "GENERATING":
            continue

        new_history_list.append(neo_history)

    return new_history_list

def get_short_memory(
        history_list: list,
        limit:int =3
):
    if limit < 1:
        raise ValueError("搜索前文数量必须大于1")

    histories: list[str] = []
    for history in history_list:

        query = str(history.get("query", history.get("question", ""))).strip()
        answer = str(history.get("answer", "")).strip()
        status = str(history.get("status", "")).strip().upper()

        if not answer:
            continue
        if status and status != "COMPLETED":
            continue

        if query:
            histories.append(f"用户：{query}\n助手：{answer}")

    if not histories:
        return ""

    recent_histories = reversed(histories[-limit:])
    return "【历史会话参考（按时间从近到远）】\n" + "\n".join(
        f"【历史会话{index}】\n{history}"
        for index, history in enumerate(recent_histories, start=1)
    )

def get_connect():
    connection_options = {
        "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.environ["MYSQL_USERNAME"],
        "password": os.environ["MYSQL_PASSWORD"],
        "charset": "utf8mb4",
        "cursorclass": cursors.DictCursor,
    }

    bootstrap_connection = pymysql.connect(**connection_options)
    try:
        with bootstrap_connection.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{DATABASE_NAME}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        bootstrap_connection.commit()
    finally:
        bootstrap_connection.close()

    return pymysql.connect(
        **connection_options,
        database=DATABASE_NAME,
    )


def retrieve_history(
        user_id:int,
        session_id:int
):
    connection = get_connect()

    try:
        with connection.cursor() as cursor:
            sql = """
            SELECT id,
                   user_id,
                   session_id,
                   task_id,
                   question,
                   answer,
                   status,
                   error_message,
                   create_time,
                   update_time
            FROM `llm_chat_table`
            WHERE user_id = %s
              AND session_id = %s
              AND status = 'COMPLETED'
              AND answer IS NOT NULL
              AND answer <> ''
            ORDER BY id ASC
            """

            cursor.execute(sql, (user_id,session_id))
            result = cursor.fetchall()

    finally:
        connection.close()

    return result


def retrieve_chat_by_id(
        chat_id:int,
        user_id:int,
        session_id:int
) -> dict | None:
    """按 Java 传入的聊天 ID 查询当前记录，不受完成状态过滤。"""
    connection = get_connect()
    try:
        with connection.cursor() as cursor:
            sql = """
            SELECT id,
                   user_id,
                   session_id,
                   task_id,
                   question,
                   answer,
                   status,
                   error_message,
                   create_time,
                   update_time
            FROM `llm_chat_table`
            WHERE id = %s
              AND user_id = %s
              AND session_id = %s
            LIMIT 1
            """
            cursor.execute(sql, (chat_id, user_id, session_id))
            return cursor.fetchone()
    finally:
        connection.close()


def retrieve_completed_chat_by_id(
        chat_id:int,
        user_id:int,
        session_id:int
) -> dict | None:
    """读取已经成功提交的聊天，供持久化记忆索引使用。"""
    connection = get_connect()
    try:
        with connection.cursor() as cursor:
            sql = """
            SELECT id,
                   user_id,
                   session_id,
                   task_id,
                   question,
                   answer,
                   status,
                   error_message,
                   create_time,
                   update_time
            FROM `llm_chat_table`
            WHERE id = %s
              AND user_id = %s
              AND session_id = %s
              AND status = 'COMPLETED'
              AND answer IS NOT NULL
              AND answer <> ''
            LIMIT 1
            """
            cursor.execute(sql, (chat_id, user_id, session_id))
            return cursor.fetchone()
    finally:
        connection.close()
