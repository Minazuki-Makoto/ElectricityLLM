import redis
import json

from redis_stream_save import (
    publish_delta,
    publish_error,
    start_put,
    end_put
)
from pydantic import BaseModel
from typing import Any


class schema(BaseModel):
    query : str
    user_id : int
    session_id : int
    task_id : str

def valid_java_request(
        data : dict[str,Any]
):
    try :
        results = schema.model_validate(data)

        return {
            field : results.get(field)
            for field in results
        }

    except Exception as e:

        raise ValueError("java传入的字段有问题",e)

def push_chunks_in_redis(
        answer : dict[str,Any]
):
    if not isinstance(answer,dict):
        raise ValueError("python产生的响应格式不符合要求")
    task_id = answer.get("task_id")
    if task_id.strip() is None:
        raise ValueError("task_id 不能为空")

    message = answer.get("message")
    if not isinstance(message,dict):
        raise ValueError("python产生的响应格式不符合要求")
    final_answer = message.get("final_answer")
    if not isinstance(final_answer,str) or final_answer.strip() is None:
        raise ValueError("python返回结果为空")

    metadata = {
        "user_id" : answer.get("user_id"),
        "session_id" : answer.get("session_id")
    }
    try:
        start_put(
            task_id,
            "",
            metadata
        )

        count = 0
        for chunk in split_chunk(final_answer, 40):
            count += 1
            publish_delta(
                task_id,
                count,
                chunk,
                metadata
            )

        end_put(
            task_id,
            "",
            count,
            metadata
        )

    except Exception as e:
        raise ValueError("redis存入过程中出现问题:",e)

def split_chunk(
        text:str,
        size:int
) -> list[str]:
    if size <= 0:
        raise ValueError("切取字段长度不能小于等于0")

    text = text.strip()

    split_result = []
    for i in range(0, len(text), size):
        end = min(i+size,len(text))
        split_result.append(text[i:end])

    return split_result


