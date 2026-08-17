# ElectricityLLM 单机 Docker Compose 部署

目标：阿里云 ECS，Ubuntu 24.04 x86_64，4 vCPU / 16 GiB RAM，无 GPU，个人低并发展示。

## 1. 部署前说明

- 公网仅开放 `22`、`80`、`443`；SSH 端口应在阿里云安全组中限制管理员 IP。
- Java、Python、MySQL、Redis、ES、Neo4j、MinIO 和 RocketMQ 均不映射公网端口。
- Python只有一个 Gunicorn worker，避免重复加载 BGE-M3 和 Reranker。
- 模型不会进入 Git 或 Docker镜像，必须挂载 `/srv/electricity-llm/models:/models:ro`。
- 当前 Nginx模板先提供 HTTP 80。绑定正式域名前，应增加证书挂载和 443 server block，或由阿里云 SLB/CDN 终止 TLS。

## 2. 服务器初始化

安装 Docker Engine 和 Compose 插件后验证：

```bash
docker version
docker compose version
```

创建 4 GiB swap（已有 swap 时不要重复创建）：

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

Elasticsearch需要提高 mmap 数量：

```bash
echo 'vm.max_map_count=1048576' | sudo tee /etc/sysctl.d/99-electricity-llm.conf
sudo sysctl --system
```

创建持久化目录：

```bash
sudo mkdir -p /srv/electricity-llm/{models,data,logs,backup}
sudo mkdir -p /srv/electricity-llm/data/{mysql,redis,elasticsearch,minio,python-logs}
sudo mkdir -p /srv/electricity-llm/data/neo4j/{data,logs,import}
sudo mkdir -p /srv/electricity-llm/data/rocketmq/{nameserver-logs,broker-logs,broker-store,proxy-logs}

sudo chown -R 1000:1000 /srv/electricity-llm/data/elasticsearch
sudo chown -R 7474:7474 /srv/electricity-llm/data/neo4j
sudo chmod -R 0770 /srv/electricity-llm/data
sudo chmod -R 0777 /srv/electricity-llm/data/rocketmq
```

RocketMQ 官方 Docker示例也要求其日志和 store 目录可写。服务稳定后可根据实际容器 UID 收紧权限。

## 3. 上传项目和配置环境变量

```bash
cd /srv/electricity-llm
git clone <repository-url> app
cd app
cp deploy.env.example .env
nano .env
chmod 600 .env
chmod +x deploy/scripts/*.sh
```

必须替换所有 `CHANGE_ME`。生产环境不得沿用本机密码。检查至少包括：

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

不要为容器地址填写 `localhost`。Compose内部应保持：

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

## 4. 准备 CPU 模型

目标目录必须是：

```text
/srv/electricity-llm/models/bge-m3
/srv/electricity-llm/models/reranker-model
```

确认目录不是空目录：

```bash
du -sh /srv/electricity-llm/models/*
test -f /srv/electricity-llm/models/bge-m3/config.json
test -f /srv/electricity-llm/models/reranker-model/config.json
```

当前本机模型目录约为 4.27 GiB 和 2.14 GiB。不要上传本地 Qwen、LoRA、训练 checkpoint 或缓存。

## 5. 配置检查和构建

```bash
cd /srv/electricity-llm/app/deploy
docker compose --env-file ../.env config --quiet
docker compose --env-file ../.env build python-agent spring-backend nginx
```

第一次构建 Python镜像需要下载 CPU PyTorch 和 FlagEmbedding 依赖，耗时取决于服务器带宽。

## 6. 启动

首次空环境可以直接启动：

```bash
docker compose --env-file ../.env up -d
docker compose --env-file ../.env ps
```

已有本机数据时，先按 `MIGRATION.md` 恢复，再启动应用层：

```bash
docker compose --env-file ../.env up -d mysql redis elasticsearch neo4j minio rocketmq-nameserver rocketmq-broker rocketmq-proxy
# 恢复数据
docker compose --env-file ../.env up -d python-agent spring-backend nginx
```

## 7. 状态、资源和日志

```bash
docker compose --env-file ../.env ps
docker stats
docker compose --env-file ../.env logs -f python-agent
docker compose --env-file ../.env logs -f spring-backend
docker compose --env-file ../.env logs -f rocketmq-proxy
```

初始资源限制：

| 服务 | 主要限制 |
|---|---:|
| Python Agent | 8 GiB、2 CPU、1 worker |
| Elasticsearch | 2 GiB容器、1 GiB heap |
| Neo4j | 2 GiB容器、768 MiB max heap、512 MiB page cache |
| Java | 768 MiB容器、512 MiB max heap |
| RocketMQ Broker | 768 MiB容器、512 MiB max heap |
| Redis | 512 MiB容器、384 MiB maxmemory、noeviction |

如果发生 OOM，先通过 `docker stats` 和容器退出码确认责任服务，不要同时提高所有 heap。16 GiB是紧缩配置，四技能完整验证后再决定是否升级到32 GiB。

## 8. 健康检查

```bash
curl -f http://127.0.0.1/health
docker compose --env-file ../.env exec python-agent curl -f http://127.0.0.1:5000/health
docker compose --env-file ../.env exec spring-backend curl -f http://127.0.0.1:8088/internal/health
```

必须验证注册、登录、会话、Chat、RAG、Light-RAG、Plot、SSE和 Memory 时序。只有 MySQL 已提交的 `COMPLETED` Chat 能进入 `memory-history`。

## 9. SSE 超时关系

默认值：

```text
Gunicorn请求上限       300 秒
Java读取Flask          330 秒
Java SSE窗口           600 秒
Nginx代理窗口          660 秒
Redis Stream TTL       900 秒
```

从内到外逐层递增，避免外层先断开而内层仍生成。调整时必须保持这个顺序。

## 10. 停止、升级和重启

```bash
docker compose --env-file ../.env stop
docker compose --env-file ../.env start
docker compose --env-file ../.env restart python-agent spring-backend
docker compose --env-file ../.env down
```

不要随意执行：

```bash
docker compose down -v
```

本项目使用宿主机 bind mount，`-v`不会替代数据备份；误删 `/srv/electricity-llm/data` 仍会造成真实数据丢失。

升级前：

1. 按 `MIGRATION.md` 备份；
2. `git pull`；
3. `docker compose build`；
4. 分应用重启；
5. 执行最小验收。

## 11. HTTPS与安全组

- 阿里云安全组只开放 80/443；22限制管理员 IP。
- 不开放 8088、5000、3306、6379、9200、7474、7687、9000、9001 和 RocketMQ端口。
- 使用 Certbot、阿里云证书服务或 SLB/CDN配置 HTTPS。
- `.env` 权限保持 `600`，不得进入日志、工单、Git和镜像。
- 用户密码当前仍存在明文兼容风险；正式公开注册前应按 `MIGRATION.md` 中的兼容哈希迁移方案处理。
