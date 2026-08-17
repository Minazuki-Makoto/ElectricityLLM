# ElectricityLLM

ElectricityLLM 是一个面向电力、电气工程知识场景的全栈智能问答系统，支持普通对话、文档 RAG、Neo4j 知识图谱检索、结构化数据绘图和分层会话记忆。

项目采用 Vue + Spring Boot + Flask/LangGraph 架构。在线文本生成使用智谱 GLM API，本地只在 CPU 上运行 BGE-M3 Embedding 和 BGE Reranker，因此部署服务器不需要 GPU。

## 系统架构

```text
Browser
   │ HTTP / HTTPS / SSE
   ▼
Nginx :80/:443
   ├─ /        → Vue 静态文件
   └─ /ai      → Spring Boot :8088
                       ├─ MySQL
                       ├─ Redis
                       ├─ RocketMQ 5.x Proxy
                       ├─ MinIO
                       └─ Flask / LangGraph :5000
                                  ├─ GLM API
                                  ├─ Elasticsearch
                                  ├─ Neo4j
                                  ├─ MySQL / Redis / MinIO
                                  └─ CPU BGE-M3 / Reranker
```

一次聊天的主要流程：

```text
前端提交问题
  → Java 创建 GENERATING 聊天记录
  → RocketMQ 异步任务
  → Python LangGraph 推理
  → Redis Stream start/delta/done/error
  → Java SSE 转发
  → Java 事务写入 COMPLETED/FAILED
  → 事务提交后调用 Python /memory/index
  → 只有 COMPLETED 回答进入 memory-history
```

## 项目目录

```text
ElectricityLLM/
├─ front/                  Vue 3 + TypeScript + Vite 前端
├─ llm-back/               Java 17 + Spring Boot 后端
├─ ElectricityLLM/         Flask + LangGraph Python Agent
├─ deploy/                 Docker Compose、Nginx及迁移工具
├─ LLM-data/               本地模型和数据，不进入Git
├─ deploy.env.example      无秘密环境变量模板
├─ DEPLOYMENT.md           服务器部署说明
├─ MIGRATION.md            数据迁移说明
└─ README.md
```

## 核心能力

- `chat`：普通对话、解释、改写和总结；
- `rag`：基于 Elasticsearch `electricity-infos` 的文档检索问答；
- `light-rag`：ES实体对齐与Neo4j多跳路径查询，证据不足时回退普通RAG；
- `plot`：从 `electricity-plot-data` 检索结构化数据，生成图表并上传MinIO；
- `memory`：Redis短期记忆、MySQL摘要与用户画像、ES历史向量记忆；
- Redis Stream + SSE：实时增量展示模型回答。

## 模型配置

| 用途 | 默认模型 | 环境变量 |
|---|---|---|
| 最终回答、复杂推理 | `glm-5.2` | `GLM_MODEL_NAME` |
| 重要路由和语义判断 | `glm-4.5-air` | `GLM_IMPORTANT_SMALL_MODEL_NAME` |
| 普通抽取和轻量分析 | `glm-4.7-flashx` | `GLM_SMALL_MODEL_NAME` |
| 向量编码 | BGE-M3，CPU | `EMBEDDING_MODEL_PATH` |
| 结果重排 | BGE Reranker，CPU | `RERANKER_MODEL_PATH` |

生产环境通过只读卷挂载模型：

```text
/srv/electricity-llm/models/bge-m3
/srv/electricity-llm/models/reranker-model
```

模型不会被复制进Docker镜像或Git仓库。

## 外部服务

| 服务 | 用途 |
|---|---|
| MySQL | 用户、会话、聊天、摘要、图表元数据、用户画像 |
| Redis | Token、缓存、限流、锁、任务状态、短期记忆、回答事件流 |
| RocketMQ | Java异步推理任务队列 |
| Elasticsearch | 文档、绘图、图谱实体和历史记忆检索 |
| Neo4j | 电力实体关系与多跳路径 |
| MinIO | 图表和文档对象存储 |
| GLM API | 在线文本生成与分析 |

MySQL六张核心表统一使用 `${MYSQL_DATABASE}`，默认数据库为 `scms`：

```text
llm_user_table
llm_session_table
llm_chat_table
plot_info
llm_chat_abstract_table
users_infos
```

## 环境变量

复制示例文件：

```bash
cp deploy.env.example .env
```

必须设置的秘密包括：

```text
GLM_API_KEY
MYSQL_ROOT_PASSWORD
MYSQL_USERNAME
MYSQL_PASSWORD
REDIS_PASSWORD
ELASTICSEARCH_PASSWORD
NEO4J_PASSWORD
MINIO_ACCESS_KEY
MINIO_SECRET_KEY
```

真实 `.env` 不得提交Git。

Docker容器之间使用服务名：

```text
MYSQL_HOST=mysql
REDIS_HOST=redis
ELASTICSEARCH_URIS=http://elasticsearch:9200
NEO4J_URI=bolt://neo4j:7687
MINIO_ENDPOINT=minio:9000
MINIO_URL=http://minio:9000
FLASK_URL=http://python-agent:5000
ROCKETMQ_ENDPOINTS=rocketmq-proxy:8081
```

注意：Python的 `MINIO_ENDPOINT` 不带协议，Java的 `MINIO_URL` 必须带协议。

## 本地开发

### Python

默认解释器：

```powershell
& "D:\anacode\python.exe" -c "import sys; print(sys.executable)"
& "D:\anacode\python.exe" -m pip install -r ".\ElectricityLLM\requirements.txt"
& "D:\anacode\python.exe" ".\ElectricityLLM\app.py"
```

### Java

```powershell
cd .\llm-back
.\mvnw.cmd spring-boot:run
```

编译验证：

```powershell
.\mvnw.cmd -DskipTests compile
```

### 前端

```powershell
cd .\front
npm.cmd ci
npm.cmd run dev
```

生产构建：

```powershell
npm.cmd run build
```

## Docker部署

目标环境：Ubuntu 24.04 LTS x86_64、无GPU。当前Compose针对4 vCPU / 16 GiB低并发展示服务器设置了紧缩资源限制。

```bash
cp deploy.env.example .env
# 修改全部CHANGE_ME

cd deploy
docker compose --env-file ../.env config --quiet
docker compose --env-file ../.env build
docker compose --env-file ../.env up -d
docker compose --env-file ../.env ps
docker stats
```

公网只应开放：

```text
22   SSH，仅管理员IP
80   HTTP
443  HTTPS
```

MySQL、Redis、ES、Neo4j、MinIO、RocketMQ、Java和Python端口不得直接暴露公网。

完整服务器初始化、目录权限、Swap、内存限制、健康检查和HTTPS说明见 [DEPLOYMENT.md](DEPLOYMENT.md)。

## 数据迁移

`deploy/scripts/` 提供：

- MySQL PowerShell/Linux导出与Linux导入；
- Elasticsearch settings、mapping、document ID和document source迁移；
- Neo4j dump/load；
- MinIO `mc mirror`；
- 数据数量验证。

详细操作见 [MIGRATION.md](MIGRATION.md)。

## 当前验证状态

已通过：

- Python在线源码AST语法检查；
- Java Maven编译；
- Vue TypeScript检查和Vite生产构建；
- `docker compose config`；
- 在线Windows绝对路径扫描；
- 暂存文件和秘密文件检查。

完整Docker镜像构建曾因本机无法连接Docker Hub OAuth而未完成，需要在网络正常环境重新执行。服务器正式开放前还必须验证四条技能、SSE事件、数据恢复、Memory写入时序和Compose重启持久化。

## 安全说明

- 不提交 `.env`、API Key、密码、Token、证书或模型；
- 不对公网开放数据库和中间件；
- 不使用Root数据库账户运行应用；
- 不在日志打印Authorization Header和秘密；
- 用户密码目前仍有旧明文兼容问题，公网开放注册前应实施渐进式BCrypt迁移；
- 执行删除索引、数据库、对象或 `docker compose down -v` 前必须备份。

## License

当前仓库尚未声明开源许可证。如计划公开发布，请先添加合适的 `LICENSE`。
