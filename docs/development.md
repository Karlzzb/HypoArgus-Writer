# 开发文档（简明）

面向本仓库开发者的入门与日常开发指引。
产品需求以根目录 `PRD.md` 为唯一事实源，领域术语见 `CONTEXT.md`，对外接口契约见 `docs/api.md`。

## 1. 项目定位

纯 LangGraph（1.x）单一技术栈的结构化写作后端服务，只提供后端能力。
核心链路：章节 → 论点 → 假说三层论证体系，5 个主节点刚性流水线 + 2 个子智能体（本期打桩），人工审阅无限迭代闭环。
LangGraph 以纯库形态嵌入自建 FastAPI，不使用 LangGraph Agent Server。

## 2. 代码布局

| 路径 | 职责 |
|---|---|
| `src/domain/` | 领域模型：图状态与状态机（`state.py`）、书目渲染、角标对账、运行单元名册、事件钩子契约 |
| `src/llm/` | 统一 LLM 封装：按运行单元前缀配置 + 全局回落（`llm_config.py`）、OpenAI 兼容客户端与 FakeLLM、Langfuse 可观测 |
| `src/assembly/` | 上下文装配：`assemble(state, unit)` 统一入口与压缩阈值配置 |
| `src/agents/` | 子智能体：任务包契约（`contracts.py`）、检索与改写打桩实现 |
| `src/nodes/` | 5 个主节点：framework_orchestrator → reference_orchestrator → writing_orchestrator → citation_validator → human_review_gate |
| `src/graph.py` | 图接线、条件路由、Postgres checkpointer |
| `src/service/` | 对外服务：FastAPI 应用（`app.py`）、任务生命周期（`task_service.py`）、事件枢纽与事件信封 |
| `docs_templates/` | 本地模板库（品类识别与大纲骨架来源） |
| `scripts/demo.py` | 全流程演示脚本（空转 / `--real` 两模式） |
| `tests/` | 按 src 分包镜像 + `tests/e2e/`（最高接缝为 `test_api_e2e.py`） |

## 3. 环境准备

- Python 3.11（conda 环境 `HypoArgus`），依赖唯一事实源为 `pyproject.toml`，用 uv 安装。
- 复制 `.env.example` 为 `.env` 并填写。

环境变量分组：

| 组 | 变量 | 说明 |
|---|---|---|
| LLM 全局缺省 | `LLM_MODEL` / `LLM_BASE_URL` / `LLM_API_KEY` | OpenAI 兼容端点；base_url 自动剥掉 `/chat/completions` 后缀 |
| LLM 单元覆盖 | `<前缀>_LLM_MODEL` / `_LLM_BASE_URL` / `_LLM_API_KEY` | 前缀共 7 个：`FRAMEWORK_ORCHESTRATOR`、`REFERENCE_ORCHESTRATOR`、`WRITING_ORCHESTRATOR`、`CITATION_VALIDATOR`、`HUMAN_REVIEW_GATE`、`SEARCH_AGENT`、`REWRITER_LOOP`；逐字段回落全局 |
| 持久化 | `HYPOARGUS_PG_DSN` | 生产 Postgres DSN，checkpointer 自动建表 |
| 可观测（可选） | `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` / `LANGFUSE_TIMEOUT` | 公私钥齐备即启用 |
| 业务调节 | `FRAMEWORK_MAX_POINTS_PER_CHAPTER`(4)、`FRAMEWORK_MAX_HYPOTHESES_PER_POINT`(3)、`FRAMEWORK_MAX_HYPOTHESES_TOTAL`(60)、`CITATION_MAX_RETRIES`(2)、`ASSEMBLER_*` 四项 | 括号内为缺省值 |
| 测试 | `HYPOARGUS_TEST_PG_DSN` | 缺省 `postgresql://postgres:postgres@127.0.0.1:15432/postgres`（本地 docker 容器 `hypoargus-test-pg`） |

## 4. 启动与演示

```bash
# 启动服务（生产路径：Postgres checkpointer）
uvicorn --app-dir src --factory service.app:create_app --host 0.0.0.0 --port 8000

# 全流程演示：假 LLM + InMemorySaver 空转
python scripts/demo.py

# 真实链路：真 LLM + Postgres + Langfuse
python scripts/demo.py --real
```

## 5. 测试与质量

```bash
python -m pytest          # pythonpath=src、testpaths=tests 已在 pyproject 配置
python -m mypy src
```

- `tests/e2e/test_api_e2e.py` 是最高接缝：httpx ASGI 全闭环（创建 → 双 SSE → 审阅 → 定稿 → 断点续跑 → 回滚）。
- `tests/e2e/test_graph_e2e.py` 依赖 Postgres（`HYPOARGUS_TEST_PG_DSN`），不可达时自动 skip。
- 离线确定性：FakeLLM + 预置应答计划（`tests/llm_response_plans.py`）；`tests/conftest.py` 会话级剥离 `LANGFUSE_*`。

## 6. 关键设计约束

- **改 `agents/` 前必读** `docs/adr/0001-subagent-real-impl-constraints.md`，遵守四条硬约束：章级落 checkpoint、事件带上下文与进度、保持非子图边界、中断场景测试。
- 节点内用 `asyncio.run` 调子智能体，因此图运行必须经 `asyncio.to_thread` 在独占工作线程同步驱动，绝不能跑在服务事件循环上（见 `src/service/task_service.py` 模块注释）。
- 双 SSE 通道严格隔离：业务通道每任务一个枢纽、终态后关闭；graph_event 可视化通道全局一个枢纽、永不主动关闭。
- 事件信封与 `state_snapshot` 只携带元数据，正文全文绝不入信封。
- 人工审阅用 `langgraph.types.interrupt` 真实中断；恢复值契约为 `{"action": "finalize"}` 或 `{"action": "revise", "feedback": "..."}`；契约不符时不抛异常，携 `error` 字段重新中断（安全汇点循环）。
- 引文内容存于 State 引文库，书目格式在交付时指定，两者完全解耦。
- ulmen-langgraph 实验结论为**不启用**（见 README 与 `tests/e2e/test_ulmen_serde_experiment.py`）。

## 7. 文档与协作约定

- 全部文档与注释使用平实中文术语；长 Markdown 每个完整句子独立成行。
- 架构决策进 `docs/adr/`；领域词汇表维护在根 `CONTEXT.md`。
- Issues 存于 GitHub（`Karlzzb/HypoArgus-Writer`），用 `gh` CLI 操作，标签规范见 `docs/agents/triage-labels.md`。
