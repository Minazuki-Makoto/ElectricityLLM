import os
import re

import pymysql
from pymysql import cursors


ALLOWED_UPDATE_FIELDS = {
    "user_true_name",
    "major",
    "profession",
    "interests",
    "prefer_plot_style",
}

DATABASE_NAME = os.getenv("MYSQL_DATABASE", "scms")


def _validate_identifier(identifier: str) -> str:
    """校验无法通过 SQL 参数占位符传入的表名和字段名。"""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
        raise ValueError(f"非法 SQL 标识符: {identifier}")
    return identifier


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


class user_info:
    id: int
    user_id: int
    user_true_name: str
    major: str
    profession: str
    interests: str
    prefer_plot_style: str


def create_table(
        connection: pymysql.Connection,
        table_name: str
) -> bool:
    table_name = _validate_identifier(table_name)
    try:
        with connection.cursor() as cursor:
            sql = f"""
            CREATE TABLE IF NOT EXISTS `{table_name}` (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                user_id BIGINT NOT NULL UNIQUE,
                user_true_name VARCHAR(255) NULL,
                major TEXT NULL,
                profession TEXT NULL,
                interests TEXT NULL,
                prefer_plot_style TEXT NULL
            )
            """
            cursor.execute(sql)
        connection.commit()
        return True
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def get_user_info(
        connection: pymysql.Connection,
        user_id: int,
        table_name: str
) -> dict:
    table_name = _validate_identifier(table_name)
    try:
        with connection.cursor() as cursor:
            sql = f"""
            SELECT id,
                   user_id,
                   user_true_name,
                   major,
                   profession,
                   interests,
                   prefer_plot_style
            FROM `{table_name}`
            WHERE user_id = %s
            LIMIT 1
            """

            cursor.execute(sql, (user_id,))
            result = cursor.fetchone()

            if not result:
                return {}

            return {
                key: value
                for key, value in result.items()
                if key not in {"id", "user_id", "session_id"}
                and value is not None
            }
    finally:
        connection.close()

def upsert_user_info(
        connection: pymysql.Connection,
        table_name: str,
        user_id: int,
        updates: dict[str, str]
) -> bool:
    """用户不存在时创建画像，存在时批量更新本次明确提供的字段。"""
    table_name = _validate_identifier(table_name)

    invalid_fields = set(updates) - ALLOWED_UPDATE_FIELDS
    if invalid_fields:
        invalid_names = ", ".join(sorted(invalid_fields))
        connection.close()
        raise ValueError(f"不允许更新字段: {invalid_names}")

    normalized_updates = {
        field: value.strip()
        for field, value in updates.items()
        if isinstance(value, str) and value.strip()
    }
    if not normalized_updates:
        connection.close()
        return False

    update_fields = list(normalized_updates)
    insert_fields = ["user_id", *update_fields]
    columns = ", ".join(f"`{field}`" for field in insert_fields)
    placeholders = ", ".join(["%s"] * len(insert_fields))
    update_clause = ", ".join(
        f"`{field}` = %s"
        for field in update_fields
    )
    values = [
        user_id,
        *(normalized_updates[field] for field in update_fields),
        *(normalized_updates[field] for field in update_fields),
    ]

    sql = f"""
    INSERT INTO `{table_name}` ({columns})
    VALUES ({placeholders})
    ON DUPLICATE KEY UPDATE {update_clause}
    """

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, values)
        connection.commit()
        return True
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def delete_user_info_fields(
        connection: pymysql.Connection,
        table_name: str,
        user_id: int,
        fields: list[str]
) -> bool:
    """将已有用户画像中指定的字段清空为 NULL。"""
    table_name = _validate_identifier(table_name)
    delete_fields = list(dict.fromkeys(fields))

    invalid_fields = set(delete_fields) - ALLOWED_UPDATE_FIELDS
    if invalid_fields:
        invalid_names = ", ".join(sorted(invalid_fields))
        connection.close()
        raise ValueError(f"不允许删除字段: {invalid_names}")

    if not delete_fields:
        connection.close()
        return False

    set_clause = ", ".join(
        f"`{field}` = NULL"
        for field in delete_fields
    )

    sql = f"""
    UPDATE `{table_name}`
    SET {set_clause}
    WHERE user_id = %s
    """

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, (user_id,))
            affected_rows = cursor.rowcount
        connection.commit()
        return affected_rows > 0
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
