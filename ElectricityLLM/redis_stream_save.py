import os

import redis
import json

ttl = int(os.getenv("CHAT_STREAM_TTL_SECONDS", "300"))

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "127.0.0.1"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    db=int(os.getenv("REDIS_DB", "0")),
    password=os.getenv("REDIS_PASSWORD") or None,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=10,
)

def generate_stream_key(
        task_id : str
) -> str:
    return f"redis:stream:{task_id}"

def publish_stream_event(
    task_id : str,
    event_type : str,
    sequence: int,
    content : str,
    metadata: dict
):
    task_id = task_id.strip()
    event_type = event_type.strip()

    if not task_id:
        raise ValueError("task_id must be set")

    if sequence < 0:
        raise ValueError("sequence must be greater than or equal to 0")

    if not event_type:
        raise ValueError("event_type must be set")

    field = {
        "task_id": task_id,
        "event_type": event_type,
        "sequence": sequence,
        "content": content,
        "metadata": json.dumps(metadata, ensure_ascii=False)
    }

    pipeline = redis_client.pipeline()
    pipeline.xadd(
        generate_stream_key(task_id),
        field,
        maxlen=2000
    )
    pipeline.expire(
        generate_stream_key(task_id),
        ttl
    )
    results = pipeline.execute()

    stream_id = results[0]

    if isinstance(stream_id, bytes):
        return stream_id.decode("utf-8")

    return stream_id

def start_put(
        task_id : str,
        content : str,
        metadata: dict
):
    stream_id = publish_stream_event(
        task_id,
        "start",
        0,
        content,
        metadata
    )

    if stream_id.strip():
        return stream_id

def end_put(
        task_id : str,
        content : str,
        sequence:int,
        metadata: dict
) :
    stream_id = publish_stream_event(
        task_id,
        "done",
        sequence,
        content,
        metadata
    )
    if stream_id.strip():
        return stream_id

def publish_delta(
      task_id: str,
      sequence: int,
      content: str,
      metadata: dict
  ) -> str:

      if not content:
          raise ValueError("delta content 不能为空")

      return publish_stream_event(
          task_id=task_id,
          event_type="delta",
          sequence=sequence,
          content=content,
          metadata=metadata
      )

def publish_error(
        task_id: str,
        sequence: int,
        error_code: str,
        error_message: str,
        metadata: dict
) -> str:
    """
    表示任务处理失败。
    """

    return publish_stream_event(
        task_id=task_id,
        event_type="error",
        sequence=sequence,
        content="",
        metadata={
            **metadata,
            "error_code": str(error_code),
            "error_message": str(error_message),
        },
    )
