# 开发文档（简明）

面向本仓库开发者的入门与日常开发指引。
架构决策以 `docs/adr/` 为准，立项需求存档见 `docs/prd-archive.md`，领域术语见 `CONTEXT.md`，对外接口契约见 `docs/api.md`，部署指引见 `docs/deployment.md`。

## 1. 项目定位

纯 LangGraph（1.x）单一技术栈的结构化写作后端服务，只提供后端能力。
核心链路：章节 → 论点 → 假说三层论证体系，6 个主节点主路径流水线（检索分支在本章检索完成后立即首写）+ 3 个子智能体（rewriter_loop、search_agent、chapter_reviewer 均为真实实现，章级写→评→重写循环见 ADR-0006），人工审阅无限迭代闭环。
LangGraph 以纯库形态嵌入自建 FastAPI，不使用 LangGraph Agent Server。

## 2. 代码布局

| 路径 | 职责 |
|---|---|
| `src/domain/` | 领域模型：图状态与状态机（`state.py`）、书目渲染、角标对账、运行单元名册、事件钩子契约 |
| `src/llm/` | 统一 LLM 封装：按运行单元前缀配置 + 全局回落（`llm_config.py`）、OpenAI 兼容客户端与 FakeLLM、Langfuse 可观测 |
| `src/assembly/` | 上下文装配：`assemble(state, unit)` 统一入口与压缩阈值配置 |
| `src/agents/` | 子智能体：任务包契约（`contracts.py`）、跨子智能体共享的线程信号量限流（`concurrency.py`）、`rewriter_loop/` 真实实现子包（编排、写作 LLM 注入点、真实适配器、风格校验器、随包风格指南；打桩同包共存、可显式注入）、`search_agent/` 真实检索适配层子包（契约映射、引擎运行时边界与假实现接缝、信号量限流；打桩同包共存、可显式注入）、`chapter_reviewer/` 章级评审子包（评审编排、评审 LLM 注入点、真实适配器、分区式修订说明纯函数装配；跨包引用 rewriter_loop 的确定性校验纯函数，ADR-0006） |
| `src/search_agent/` | 检索引擎包：自源项目一次性 fork 的 SearchAgent V12（火山联网 / Bisheng 知识库 / Doris 结构化三通道），归本项目所有、可自由改造；经 `src/agents/search_agent/` 薄适配层接入主流程（LLM 配置走 `SEARCH_AGENT_LLM_*` 统一解析） |
| `src/nodes/` | 6 个主节点：framework_orchestrator → reference_orchestrator（检索并行扇出；每章检索持久化后立即首写）→ chapter_drafter（无检索章节的首写）→ writing_orchestrator（修订与回退串行自环）→ document_reviewer（篇级终审门禁）→ human_review_gate |
| `src/graph.py` | 图接线、条件路由、Postgres 检查点保存器（含 `checkpoint_serializer` 类型注册） |
| `src/service/` | 对外服务：FastAPI 应用（`app.py`）、任务生命周期（`task_service.py`）、事件枢纽与事件信封 |
| `docs_templates/` | 本地模板库（品类识别与大纲骨架来源） |
| `scripts/demo.py` | 全流程演示脚本（离线 / `--real` 两模式），每次运行落盘构建过程档案 |
| `scripts/rewriter_debug.py` | rewriter_loop 独立调测脚本（绕开主图，供 prompt 与风格规则调优） |
| `tests/` | 按 src 分包镜像 + `tests/e2e/`（最高注入点为 `test_api_e2e.py`）+ `tests/scripts/`（`scripts/` 调测脚本的冒烟测试） |

## 3. 环境准备

- Python 3.11（conda 环境 `HypoArgus`），依赖唯一事实源为 `pyproject.toml`，用 uv 安装。
- 复制 `.env.example` 为 `.env` 并填写。

环境变量分组：

| 组 | 变量 | 说明 |
|---|---|---|
| LLM 全局缺省 | `LLM_MODEL` / `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_ENABLE_THINKING` | OpenAI 兼容端点；base_url 自动剥掉 `/chat/completions` 后缀；思考模式缺省关闭（`0`） |
| LLM 单元覆盖 | `<前缀>_LLM_MODEL` / `_LLM_BASE_URL` / `_LLM_API_KEY` / `_LLM_ENABLE_THINKING` | 前缀共 9 个：`FRAMEWORK_ORCHESTRATOR`、`REFERENCE_ORCHESTRATOR`、`CHAPTER_DRAFTER`、`WRITING_ORCHESTRATOR`、`DOCUMENT_REVIEWER`、`HUMAN_REVIEW_GATE`、`SEARCH_AGENT`、`REWRITER_LOOP`、`CHAPTER_REVIEWER`；逐字段回落全局 |
| 持久化 | `HYPOARGUS_PG_DSN` | 生产 Postgres 连接串，检查点保存器自动建表 |
| 可观测（可选） | `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` / `LANGFUSE_TIMEOUT` / `LANGFUSE_TRACING_ENABLED` | 公私钥齐备即启用；总开关设 `false` 关闭上报（公私钥可留着不动） |
| 业务调节 | `FRAMEWORK_MAX_POINTS_PER_CHAPTER`(4)、`FRAMEWORK_MAX_HYPOTHESES_PER_POINT`(3)、`FRAMEWORK_MAX_HYPOTHESES_TOTAL`(60)、`DOCUMENT_REVIEW_MAX_RETRIES`(2)、`CHAPTER_MAX_REWRITES`(1)、`REFERENCE_SEARCH_TIMEOUT_SECONDS`(600)、`ASSEMBLER_*` 五项（含篇级终审全文段阈值 `ASSEMBLER_DOCUMENT_TEXT_MAX_CHARS`，缺省 30000） | 括号内为缺省值；`CHAPTER_MAX_REWRITES` 是章级写→评→重写循环的重写次数上限，设 0 关闭评审重写、只保留纯首写；`REFERENCE_SEARCH_TIMEOUT_SECONDS` 是单章检索硬超时，检索源停顿时该分支按超时失败而非静默挂死 |
| 并发度 | `GRAPH_MAX_CONCURRENCY`(4)、`FRAMEWORK_MAX_CONCURRENT_CHAPTERS`(4)、`DOCUMENT_REVIEW_MAX_CONCURRENT_CHAPTERS`(4)、`SEARCH_AGENT_MAX_CONCURRENT_CALLS`(6)、`CHAPTER_REVIEWER_MAX_CONCURRENT_CALLS`(2) | 依次为：图级并行分支上限（检索分支与无检索章节的首写 Send 扇出）、论证框架论点假说生成与引文语义核查的章节级 LLM 并发上限、检索运行时全局整章引擎调用预算、章级评审外部调用总并发。search_agent 预算由同一服务运行时的并行章节与增量检索共享，事件会暴露许可排队与 `queue_wait_ms`；引擎内部 provider 级限流仍独立生效（共享线程信号量机制见 `src/agents/concurrency.py`）。 |
| 检索通道 | `VOLCANO_SEARCH_*`、`BISHENG_*`、`DORIS_*`、`SEARCH_AGENT_SHADOW_MODE`、`SEARCH_AGENT_*_LLM_ENABLED`、`JUDGE_MODEL`、`EVIDENCE_RETRIEVAL_<字段>` | search_agent 检索引擎三通道接入与细项配置，逐项说明见 `.env.example` 的检索通道分节 |
| 调测 | `LLM_DEBUG_TIMING` | 设为 `1` 打印逐次 LLM 调用计时 |
| 测试 | `HYPOARGUS_TEST_PG_DSN` | 缺省 `postgresql://postgres:postgres@127.0.0.1:15432/postgres`（本地 docker 容器 `hypoargus-test-pg`） |

## 4. 启动与演示

```bash
# 启动服务（生产路径：Postgres 检查点保存器）
uvicorn --app-dir src --factory service.app:create_app --host 0.0.0.0 --port 8000

# 全流程演示（离线模式）：假 LLM + 内存检查点保存器，
# 写作走 rewriter_loop 真实实现链路，仅最底层模型是假的
python scripts/demo.py

# 真实链路：真 LLM + Postgres + Langfuse
python scripts/demo.py --real

# 每次运行落盘一份构建过程档案（事件流、逐章产物、state 演进、最终全文与书目）
# 缺省写入 var/demo_archive/<thread_id>.md，可用 --archive PATH 覆盖
# 运行到定稿时另落一份成品文档（仅重编号正文 + 参考文献），同名加 -article 后缀
python scripts/demo.py --archive out.md

# 人工门默认自动模拟（脚本按序提交 revise→confirm→finalize，保证可复现基准）；
# 传 --no-auto-review 关闭自动模拟：到达中断点时仅打印 REST 指令，
# 由人在另一终端 curl 提交，服务端续跑后本进程继续等待下一事件
python scripts/demo.py --real --no-auto-review --port 8000

# 每次人工中断点（含反馈前第一次）都会独立落盘一版初稿供人审阅：
#   <档案名>-article-r1.md  反馈前原文
#   <档案名>-article-r2.md  revise 后、finalize 前中间版
#   <档案名>-article.md     定稿最终版（同名 -article 后缀）
# 人工模式下需推到 finalize 才产出最终成品；SIGTERM 硬杀不跑 finally 会丢档案，用 Ctrl+C 仍能落盘过程档案
python scripts/demo.py --no-auto-review --archive out.md

# 性能调测：LLM_DEBUG_TIMING=1 打印逐次 LLM 调用计时
LLM_DEBUG_TIMING=1 uv run python scripts/demo.py --real
```

rewriter_loop 独立调测（绕开主图直接驱动真实实现，供 prompt 与风格规则调优）：

```bash
python scripts/rewriter_debug.py                        # 离线全流程（进度事件打印到终端）
python scripts/rewriter_debug.py --mode revise          # 定向改写模式（覆盖任务包内 mode）
python scripts/rewriter_debug.py --step lint            # 跑到指定环节停：write/lint/audit/revise-fix
python scripts/rewriter_debug.py --task scripts/rewriter_task.sample.json   # 从文件读任务包
python scripts/rewriter_debug.py --real                 # 真实模型（REWRITER_LOOP_LLM_* 配置，有花费）
```

## 5. 测试与质量

```bash
python -m pytest                 # pythonpath=src、testpaths=tests 已在 pyproject 配置
python -m mypy src scripts tests
```

- `tests/e2e/test_api_e2e.py` 是最高注入点：httpx ASGI 全闭环（创建 → 双 SSE → 审阅 → 定稿 → 断点续跑 → 回滚）。
- `tests/e2e/test_graph_e2e.py` 依赖 Postgres（`HYPOARGUS_TEST_PG_DSN`），不可达时自动 skip；其中断恢复用例走 rewriter_loop 真实实现完整链路（仅最底层 FakeLLM）。
- 离线确定性：FakeLLM + 预置应答计划（`tests/llm_response_plans.py`）；`tests/conftest.py` 会话级剥离 `LANGFUSE_*`。

## 6. 关键设计约束

- **改 `agents/` 前必读** `docs/adr/0001-subagent-real-impl-constraints.md`，遵守四条硬约束：章级落 checkpoint、事件带上下文与进度、保持非子图边界、中断场景测试。
- 首写阶段由 chapter_drafter 经 `Send` 并行扇出，各分支承接前章规划摘要链、章稿经合并 reducer 回写 state；崩溃恢复依赖 LangGraph 超步事务的 pending writes 语义，只重跑未完成分支。
- rewriter_loop 真实实现迁移的架构决策（契约零扩展、LLM 栈归一、修一次链路等）见 `docs/adr/0002-rewriter-loop-migration.md`。
- rewriter_loop 字数管控口径（三级区间、散文统计、表章豁免、修后字数复检例外）见 `docs/adr/0003-word-count-control.md`。
- `domain/state.py` 新增状态模型无需手工登记序列化白名单：`graph.py` 的 `CHECKPOINT_MSGPACK_TYPES` 自动收集该模块全部 pydantic 模型与枚举，注册进检查点序列化器（严格模式 `LANGGRAPH_STRICT_MSGPACK=true` 下往返成立，有 `tests/test_checkpoint_serde.py` 回归覆盖）。
- 节点内用 `asyncio.run` 调子智能体，因此图运行必须经 `asyncio.to_thread` 在独占工作线程同步驱动，绝不能跑在服务事件循环上（见 `src/service/task_service.py` 模块注释）。
- 双 SSE 通道严格隔离：业务通道每任务一个枢纽、终态后关闭；graph_event 可视化通道全局一个枢纽、永不主动关闭。
- 事件信封与 `state_snapshot` 只携带元数据，正文全文绝不入信封。
- 人工审阅用 `langgraph.types.interrupt` 真实中断；恢复值契约为 `{"action": "finalize"}`、`{"action": "revise", "feedback": "..."}` 或 `{"action": "confirm"}`（仅在大扇出确认中断时可用，动作全集见 `human_review_gate.RESUME_ACTIONS`）。
契约不符时不抛异常，携 `error` 字段重新中断；意见含混/定位失败携 `clarification_questions` 回问；受影响章数超过大纲一半携 `pending_confirmation` 清单待确认（均为安全汇点循环，ADR-0009）。
- 引文内容存于 State 引文库，书目格式在交付时指定，两者完全解耦。
- ulmen-langgraph 实验结论为**不启用**（见 README 与 `tests/e2e/test_ulmen_serde_experiment.py`）。

## 7. 文档与协作约定

- 全部文档与注释使用平实中文术语；长 Markdown 每个完整句子独立成行。
- 架构决策进 `docs/adr/`；领域词汇表维护在根 `CONTEXT.md`。
- Issues 存于 GitHub（`Karlzzb/HypoArgus-Writer`），用 `gh` CLI 操作，标签规范见 `docs/agents/triage-labels.md`。
