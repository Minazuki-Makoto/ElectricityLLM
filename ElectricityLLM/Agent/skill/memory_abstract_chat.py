import os

import pymysql
from pymysql import cursors
from redis import Redis
import json

DATA_BASE = os.getenv("MYSQL_DATABASE", "scms")

redis = Redis(
    host=os.getenv("REDIS_HOST", "127.0.0.1"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    db=int(os.getenv("REDIS_DB", "0")),
    password=os.getenv("REDIS_PASSWORD") or None,
    decode_responses=True,
)

def generate_abstract_save_key(user_id:int):
    return f"redis:list:{user_id}:abstract"

async def read_value(user_id:int,session_id:int,table_name:str):
    redis_key = generate_abstract_save_key(user_id)

    read_type = "redis"
    try:
        results = redis.lrange(
            redis_key,
            0,
            -1
        )

        if len(results) == 0:
            read_type = "sql"
            results = get_abstraction_info_from_sql(
                table_name,
                user_id,
                session_id
            )
            if len(results) == 0:
                return  ""

    finally:
        if read_type == "sql" and len(results) > 0:
            for result in results:
                redis.lpush(
                    redis_key,
                    json.dumps(result)
                )
        return [json.loads(result) for result in results]



def get_connect():

    connection_args = {
        "host":os.getenv("MYSQL_HOST", "127.0.0.1"),
        "port":int(os.getenv("MYSQL_PORT", "3306")),
        "user":os.environ["MYSQL_USERNAME"],
        "password":os.environ["MYSQL_PASSWORD"],
        "database":DATA_BASE,
        "charset":"utf8mb4",
        "cursorclass":cursors.DictCursor
    }

    return pymysql.connect(**connection_args)

def create_table(
    abstract_table_name:str
):

    connection = get_connect()

    try:
        with connection.cursor() as cursor:
            sql = f"""
            CREATE TABLE IF NOT EXISTS `{abstract_table_name}`(
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                user_id BIGINT NOT NULL,
                session_id BIGINT NOT NULL,
                chat_start_end TEXT NOT NULL,
                abstract TEXT NOT NULL,
                create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """

            cursor.execute(sql)
        connection.commit()

    finally:
        connection.close()


#五个一条，进行摘要总结，写入
def write_in_sql(
        abstract_table_name:str,
        user_id:int,
        session_id:int,
        chat_start_end:str,
        abstract
):
    connection = get_connect()

    try:

        with connection.cursor() as cursor:
            sql = f"""
            INSERT INTO `{abstract_table_name}`(user_id, session_id, chat_start_end, abstract)
            VALUES (%s, %s, %s, %s)
            """

            cursor.execute(sql, (user_id, session_id, chat_start_end, abstract))
            abstract_id = cursor.lastrowid
        connection.commit()
        return abstract_id

    finally:
        connection.close()


def get_abstraction_info_from_sql(
        abstract_table_name:str,
        user_id:int,
        session_id:int,
):
    connection = get_connect()
    try:
        with connection.cursor() as cursor:

            sql = f"""
            SELECT id,
                   user_id,
                   session_id,
                   chat_start_end,
                   abstract,
                   create_time,
                   update_time
            FROM `{abstract_table_name}`
            WHERE user_id = %s AND session_id = %s
            """

            cursor.execute(sql,(user_id,session_id))
            result = cursor.fetchall()

    finally:
        connection.close()

    return result

def write_in_redis(
        user_id:int,
        session_id:int,
        chat_start_end:str,
        abstract
):
    redis_key = generate_abstract_save_key(user_id)
    content = {
        "user_id":user_id,
        "session_id":session_id,
        "chat_start_end":chat_start_end,
        "abstract":abstract
    }

    try:
        redis.lpush(
            redis_key, json.dumps(content)
        )

        return True
    except Exception as e:

        return False
