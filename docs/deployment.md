# 部署文档（面向运维）

本文面向负责部署与运维 HypoArgus-Writer 的人员，只覆盖装机、配置、启动、巡检与故障恢复。
接口契约见 `docs/api.md`；开发相关内容见 `docs/development.md`。

## 1. 系统要求

| 项 | 要求 |
|---|---|
| 操作系统 | Linux（x86_64） |
| Python | 3.11 |
| 数据库 | PostgreSQL 14 及以上，服务可网络直连 |
| 出网 | 能访问所配置的大模型接口地址（如阿里云百炼 `dashscope.aliyuncs.com`）；启用 Langfuse 时还需可达其地址 |

服务是无状态进程：全部任务状态持久化在 PostgreSQL 检查点表中，进程重启不丢数据。

## 2. 安装（含国内网络渠道建议）

依赖清单以 `pyproject.toml` 为唯一事实源，使用 uv 安装。
国内网络直连官方 PyPI 与 GitHub 较慢，建议按以下渠道安装。

### 2.1 安装 uv

官方安装脚本走 GitHub，国内经常超时，推荐直接从清华 PyPI 镜像装：

```bash
pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2.2 配置 PyPI 镜像

任选其一（推荐环境变量方式，写进部署用户的 `~/.bashrc`）：

```bash
# 清华镜像（也可换阿里云 https://mirrors.aliyun.com/pypi/simple/）
export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
```

注意：uv 不继承 pip 的镜像配置，必须单独设置。
若需要 uv 代管 Python 解释器（本机无 3.11 时），解释器安装包在 GitHub 上，需另设：

```bash
export UV_PYTHON_INSTALL_MIRROR="https://mirrors.tuna.tsinghua.edu.cn/github-release/astral-sh/python-build-standalone/"
```

### 2.3 安装项目依赖

```bash
cd HypoArgus-Writer
uv sync          # 按 uv.lock 精确安装，可重复执行
```

### 2.4 PostgreSQL

有现成实例直接用即可，无版本外的特殊要求。
自建时推荐发行版软件源（`apt install postgresql`）或公司内部镜像仓库的官方镜像；
直连 Docker Hub 在国内不稳定，请先给 Docker 配置可用的镜像加速器再拉取。

## 3. 配置

复制仓库根的 `.env.example` 为 `.env`，放在服务工作目录。
必填三组：

| 变量 | 说明 |
|---|---|
| `LLM_MODEL` / `LLM_BASE_URL` / `LLM_API_KEY` | 大模型全局配置，OpenAI 兼容接口；`LLM_BASE_URL` 止于兼容根路径（不要带 `/chat/completions`） |
| `HYPOARGUS_PG_DSN` | PostgreSQL 连接串，形如 `postgresql://user:pass@host:5432/dbname` |
| `LANGFUSE_*`（可选） | 公私钥齐备即启用调用链上报，不配置则完全关闭 |

其余可选项（各环节独立模型、并发度、业务上限等）见 `.env.example` 内注释，缺省值即可运行。
数据库表由服务首次启动时自动创建，无需手工建表；库账号需要建表权限。

## 4. 启动

```bash
uvicorn --app-dir src --factory service.app:create_app --host 0.0.0.0 --port 8000
```

生产建议交给 systemd 托管，示例单元文件：

```ini
[Unit]
Description=HypoArgus-Writer
After=network-online.target postgresql.service

[Service]
User=hypoargus
WorkingDirectory=/opt/HypoArgus-Writer
ExecStart=/opt/HypoArgus-Writer/.venv/bin/uvicorn --app-dir src --factory service.app:create_app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

`uv sync` 会在项目根创建 `.venv`，`ExecStart` 直接用其中的 `uvicorn` 即可，无需激活虚拟环境。

### 启动后验证

```bash
curl -sf http://127.0.0.1:8000/openapi.json > /dev/null && echo OK
```

返回 OK 即服务就绪（该地址同时可作为存活探测端点）。
注意：接口无鉴权，端口只应暴露给内网调用方，不要直接暴露公网。

## 5. 日常运维

- **日志**：uvicorn 标准输出，systemd 下用 `journalctl -u hypoargus-writer -f` 查看。
- **备份**：全部业务状态都在 PostgreSQL 中，常规库备份即覆盖全量任务数据；服务本身无本地状态文件。
- **升级**：`git pull` 后执行 `uv sync` 再重启服务即可；数据库结构由 LangGraph 自动迁移。
- **调用链观测**：配置了 Langfuse 时，每次任务一条调用链，可按 `thread_id` 检索每次模型调用的输入输出、token 用量与耗时。

## 6. 故障恢复

服务重启后内存中的任务登记会丢失，但检查点仍在 PostgreSQL 中。
对重启前未完成的任务，由调用方（或运维手工）触发恢复：

```bash
curl -X POST http://127.0.0.1:8000/tasks/<thread_id>/resume \
  -H 'Content-Type: application/json' -d '{}'
```

三种结果均为正常：停在人工审阅点的任务补发提醒事件；中途被杀的任务从最近检查点继续；已完成的任务仅重建登记。
更多接口语义见 `docs/api.md` 2.4 节。

## 7. 常见问题

- **启动即报「缺少 Postgres 连接串」**：`.env` 未放在工作目录或未配置 `HYPOARGUS_PG_DSN`。
- **模型调用超时或 401**：核对 `LLM_BASE_URL` 与 `LLM_API_KEY`；国内环境建议选用国内大模型服务商的 OpenAI 兼容端点。
- **修改 Langfuse 配置不生效**：是否启用在服务启动时确定，改配置后需重启服务。
