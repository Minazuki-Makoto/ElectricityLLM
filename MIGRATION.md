# ElectricityLLM 数据迁移手册

迁移顺序：MySQL → Elasticsearch → Neo4j → MinIO。任何删除、覆盖或恢复操作前必须保留第二份离线备份。

## 1. MySQL

本机权威数据库已经统一为 `${MYSQL_DATABASE}`，默认 `scms`。六张表是：

```text
llm_user_table
llm_session_table
llm_chat_table
plot_info
llm_chat_abstract_table
users_infos
```

其中本机审计时前五张表已存在；`users_infos` 尚未在 `scms` 创建，服务器 baseline 按当前 Python字段建立，之后 Python 与前五张表共用同一数据库。

Linux导出：

```bash
set -a; source .env; set +a
./deploy/scripts/export_mysql.sh ./backup/mysql/scms.sql
```

服务器导入：

```bash
set -a; source .env; set +a
MYSQL_HOST=127.0.0.1 ./deploy/scripts/import_mysql.sh ./backup/mysql/scms.sql
```

如果从宿主机无法访问未映射端口，使用容器：

```bash
docker compose --env-file .env -f deploy/docker-compose.yml exec -T mysql \
  sh -c 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD"' < backup/mysql/scms.sql
```

空数据库会自动执行 `deploy/mysql/init/schema.sql`。该目录只在 MySQL数据目录首次初始化时执行；已有数据卷不会重复执行。

### 密码哈希迁移

当前 `llm_user_table.password` 存在明文账号。不要直接批量把未知明文替换为哈希并同时修改登录逻辑。建议独立实施兼容迁移：

1. 新注册用户只写 BCrypt；
2. 登录时若值以 BCrypt前缀开头则验证哈希；
3. 若是旧明文且验证成功，则在同一用户记录中原位升级为 BCrypt；
4. 更新/删除用户也通过统一 PasswordService验证；
5. 所有旧活跃账号升级后，再移除明文兼容分支。

## 2. Elasticsearch

Windows本机导出：

```powershell
$env:ELASTICSEARCH_URIS = "https://localhost:9200"
& "D:\anacode\python.exe" `
  "D:\pycharmcode\ElectricityLLM\deploy\scripts\elasticsearch_migrate.py" `
  export --directory "D:\CodexWork\outputs\electricity-es-backup"
```

脚本为每个索引生成：

```text
<index>.meta.json       settings + mapping
<index>.docs.ndjson     _id + _source
```

服务器导入：

```bash
set -a; source .env; set +a
python3 deploy/scripts/elasticsearch_migrate.py import --directory backup/elasticsearch
python3 deploy/scripts/elasticsearch_migrate.py verify
```

必须比较四个索引的 document count：

```text
electricity-infos
electricity-plot-data
graph_index
memory-history
```

## 3. Neo4j

本机审计结果：4,556个节点、1,363条关系，没有用户自定义 constraint 或 property index，仅有系统 lookup indexes。

Community Edition一致性 dump 需要短暂停机：

```bash
./deploy/scripts/export_neo4j.sh
./deploy/scripts/import_neo4j.sh /path/to/neo4j.dump
```

恢复后执行：

```cypher
MATCH (n) RETURN count(n);
MATCH ()-[r]->() RETURN count(r);
SHOW CONSTRAINTS;
SHOW INDEXES;
```

`deploy/neo4j/schema.cypher` 是后续 schema版本入口。没有确认实际 label唯一性前，不得擅自添加 constraint。

## 4. MinIO

安装 MinIO Client `mc` 后设置源和目标变量，不要把密钥写进命令历史或脚本：

```bash
export SOURCE_MINIO_URL=http://source-host:9000
export SOURCE_MINIO_ACCESS_KEY=...
export SOURCE_MINIO_SECRET_KEY=...
export TARGET_MINIO_URL=http://target-host:9000
export TARGET_MINIO_ACCESS_KEY=...
export TARGET_MINIO_SECRET_KEY=...
./deploy/scripts/mirror_minio.sh
```

脚本迁移 `plot` 和 `contracts`，桶名可由 `MINIO_PLOT_BUCKET`、`MINIO_DOCUMENT_BUCKET`覆盖。

## 5. 重启持久化验收

```bash
docker compose --env-file .env -f deploy/docker-compose.yml restart
docker compose --env-file .env -f deploy/docker-compose.yml ps
./deploy/scripts/verify_data.sh
```

确认 MySQL记录、ES索引、Neo4j节点关系、MinIO对象和模型挂载均未丢失，再开放公网访问。
