# HypoArgus-Writer

纯 LangGraph 单一技术栈的工业级结构化写作后端服务，只提供后端能力，不含前端页面。
立项需求存档见 `docs/prd-archive.md`；领域术语以 `CONTEXT.md` 词汇表为准。

## 架构总览

由 6 个 LangGraph 主节点组成主路径流水线，构成迭代闭环：

```
framework_orchestrator → reference_orchestrator → chapter_drafter → writing_orchestrator → document_reviewer → human_review_gate
     论证框架生成          并行检索（Send 扇出）   并行首写（Send 扇出）    修订与回退串行总控        篇级终审门禁          人工中断点与迭代路由
```

检索与首写两段均经 `Send` 并行扇出：reference_orchestrator 每个待检索章节一个分支、只回写单章素材，引文库经合并 reducer 汇入并跨章按 URL 去重；chapter_drafter 每个未写章节一个分支，各分支承接前章规划摘要链、只回写单章草稿；修订与终审回退仍由 writing_orchestrator 串行自环处理。

- 论证体系三层结构：章节 1—n 论点，论点 1—N 假说；假说可证伪、可检索验证，是检索任务的直接驱动源。
- 三个业务子智能体 search_agent（检索与素材相关性校验）、rewriter_loop（章节写作）、chapter_reviewer（章级评审）以黑盒适配层接入，均为真实实现且是缺省装配。
search_agent 经薄适配层调用 fork 进本项目的 SearchAgent V12 检索引擎（火山联网 / Bisheng 知识库 / Doris 结构化三通道），打桩同包保留供测试注入。
检索过程逐检索项上报进度事件（正反向检索、通道调用、证据裁决 x/y、补漏轮次），经既有事件通道进 SSE；引擎诊断计数/耗时全量进 Langfuse span，摘要子集随子智能体结束事件上报。
- 评审分两级：chapter_reviewer 章级评审只裁单章内部质量；document_reviewer 篇级终审在引用核查与结构完整性之上做一次全篇 LLM 评审——跨章硬事实冲突为 error 自动打回，章间衔接 / 口径统一 / 跨章重复为 warn 随人工中断点呈现、不打回（雪崩防护），error/warn 裁决权在代码不在模型。
- 引用采用「正文角标 + 结构化引文库」分离方案；最终交付可按任意书目格式渲染，格式与内容解耦。
- human_review_gate 是全流程唯一安全汇点：任何机器环节失败若干次后都塌缩到这里，系统永不卡死。
- 全流程状态经 LangGraph 官方 Postgres 检查点保存器（checkpointer）持久化，支持断点续跑与历史版本回滚。

## 环境准备

- Python 3.11（既有 conda 环境），依赖以 `pyproject.toml` 为唯一事实源，uv 作为安装器。
- Postgres（生产运行必需，检查点保存器建表自动完成）。
- 自建 Langfuse（可选，配置后自动上报全链路 LLM 调用）。

```bash
conda activate HypoArgus
uv sync                       # 安装运行时与 dev 依赖
uv sync --group experiment    # 可选：ulmen serde 实验复现用
cp .env.example .env          # 按下文约定填写
```

## `.env` 配置约定

完整示例见 `.env.example`，要点如下。

### LLM 配置（单元前缀 + 全局回落）

全部 9 个运行单元支持独立配置，未配置项逐字段回落无前缀的全局缺省变量。

| 变量 | 说明 |
| --- | --- |
| `LLM_MODEL` / `LLM_BASE_URL` / `LLM_API_KEY` | 全局缺省，必填 |
| `<单元前缀>_LLM_MODEL` 等三项 | 单元独立配置，可选 |

单元前缀共 9 个：`FRAMEWORK_ORCHESTRATOR`、`REFERENCE_ORCHESTRATOR`、`CHAPTER_DRAFTER`、`WRITING_ORCHESTRATOR`、`DOCUMENT_REVIEWER`、`HUMAN_REVIEW_GATE`、`SEARCH_AGENT`、`REWRITER_LOOP`、`CHAPTER_REVIEWER`。
所有模型统一按 OpenAI 兼容接口封装，`base_url` 止于兼容根路径（不要带 `/chat/completions`）。

### 持久化与可观测

| 变量 | 说明 |
| --- | --- |
| `HYPOARGUS_PG_DSN` | Postgres 检查点保存器连接串，生产必填 |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | 两者齐备即启用 Langfuse 上报，缺省不启用 |
| `LANGFUSE_BASE_URL` | 自建 Langfuse 接口地址 |
| `LANGFUSE_TIMEOUT` | 大 generation 事件导出超时（秒） |

### 业务调节项（均可选）

| 变量 | 缺省 | 说明 |
| --- | --- | --- |
| `FRAMEWORK_MAX_POINTS_PER_CHAPTER` | 4 | 每章论点数上限 |
| `FRAMEWORK_MAX_HYPOTHESES_PER_POINT` | 3 | 每论点假说数上限 |
| `FRAMEWORK_MAX_HYPOTHESES_TOTAL` | 60 | 全文假说总数上限 |
| `DOCUMENT_REVIEW_MAX_RETRIES` | 2 | 篇级终审失败重试上限，超限带警告交人工裁决 |
| `GRAPH_MAX_CONCURRENCY` | 4 | 图级并行分支上限（首写阶段 Send 扇出的并发度） |
| `FRAMEWORK_MAX_CONCURRENT_CHAPTERS` | 4 | 论证框架论点假说生成的章节级 LLM 并发上限 |
| `DOCUMENT_REVIEW_MAX_CONCURRENT_CHAPTERS` | 4 | 引文语义核查的章节级 LLM 并发上限 |
| `SEARCH_AGENT_MIN_PASS_PER_CHAPTER` | 3 | 每章 pass 落库下限，低于此值发薄弱章警告并计入诊断摘要（不阻断不补检） |
| `ASSEMBLER_*` | 见 `.env.example` | 上下文装配压缩阈值与保留策略 |

## 启动方式

```bash
uvicorn --app-dir src --factory service.app:create_app --host 0.0.0.0 --port 8000
```

主要接口：

| 接口 | 说明 |
| --- | --- |
| `POST /tasks` | 创建写作任务（`user_intent`、`user_identity`、`session_id`） |
| `GET /tasks/{thread_id}` | 任务状态摘要 |
| `POST /tasks/{thread_id}/review` | 提交人工审阅（`finalize` 定稿 / `revise` 携意见修订） |
| `POST /tasks/{thread_id}/resume` | 崩溃后按检查点恢复 |
| `POST /tasks/{thread_id}/rollback` | 回滚到指定历史检查点 |
| `GET /tasks/{thread_id}/checkpoints` | 检查点元数据清单 |
| `GET /tasks/{thread_id}/bibliography?format=` | 按书目格式渲染最终交付（`gbt7714` / `apa` / `markdown`） |
| `POST /retrieval` | 独立阻塞式检索：一章假说列表同步换素材与诊断，不启动写作任务 |
| `GET /tasks/{thread_id}/stream` | 业务数据 SSE 通道 |
| `GET /graph_events` | `graph_event` 可视化 SSE 通道，支持 `thread_id` / `session_id` / `types` 过滤 |

两条 SSE 通道严格隔离：业务通道只发轻量业务事件，可视化通道只发事件信封（携带父子链路 ID，供前端自主还原动态 DAG 拓扑）。

## 演示脚本

```bash
python scripts/demo.py           # 离线演示：假 LLM + 内存检查点保存器，离线可复现
python scripts/demo.py --real    # 与生产一致的演示：真实 LLM + Postgres + Langfuse（需 .env 就绪）
```

脚本驱动一遍完整闭环：创建任务 → 双 SSE 流 → 混合两类分支（纯改写 + 补充佐证）的修订迭代 → 篇级终审门禁 → 定稿 → 两种书目格式渲染。

## 可观测（Langfuse）

- 通过官方插桩接入，不使用 LangSmith。
- 本项目 LLM 封装直连 OpenAI 兼容接口而非 langchain Runnable，官方 LangChain 回调处理器捕捉不到这些调用，故 LLM 调用采用官方 `langfuse.openai` 插桩客户端上报，节点与子智能体用轻量 span 包装挂到同一条 trace。
- `LANGFUSE_PUBLIC_KEY` 与 `LANGFUSE_SECRET_KEY` 齐备即启用；未配置时完全无副作用。
- 启用与否在服务启动（构图）时确定，修改 Langfuse 配置后需重启服务。
- 每次图运行一条 trace，关联 `thread_id` / `session_id` / `execution_trace_id`。
- trace 覆盖全部 9 个运行单元：6 个主节点为 `node:*` span，3 个子智能体为 `subagent:*` span，每次 LLM 调用自动上报 generation（输入输出、token 用量、耗时、成本）。
- 人工中断点的正常中断不会被标记为错误 span。

## ulmen 压缩 serde 实验结论：不启用

立项约定 ulmen-langgraph 压缩仅作为实验性可选序列化器接入检查点保存器，且「关闭开关后历史检查点必须仍可读取，做不到则不启用」。
实验结论（`tests/e2e/test_ulmen_serde_experiment.py` 固化为回归证据）：

- 正向兼容成立：开启压缩后可以读取此前未压缩写入的历史存档。
- 反向兼容不成立：压缩写入的存档在关闭开关后，纯 PostgresSaver 读取不报错而是静默解出损坏数据（msgpack 把 ULMZ 魔数首字节当作整数解码并丢弃其余内容）。

静默数据损坏比读取报错更危险，因此本项目不启用 ulmen serde，运行时不提供接入开关。
`ulmen-langgraph` 仅保留在 `experiment` 依赖组中供复现实验；若上游修复反向兼容，该实验测试会失败，届时再重新评估接入。

## 测试

```bash
python -m pytest        # 全量测试
python -m mypy src      # 类型检查
```

- 只测外部行为，不测实现细节；最高测试注入点为 FastAPI HTTP 层的端到端主干测试。
- 第二道注入点是统一 LLM 调用封装层，注入确定性假 LLM 使主节点与真实写作链路行为可复现；打桩子智能体仍可显式注入作测试替身。
- 依赖 Postgres 的测试按 `HYPOARGUS_TEST_PG_DSN`（缺省 `postgresql://postgres:postgres@127.0.0.1:15432/postgres`）连接，不可达时自动跳过。

## 文档

- `CONTEXT.md` — 领域词汇表，全部文档与注释使用平实中文术语。
- `docs/prd-archive.md` — 立项需求存档（问题陈述、用户故事、范围外），不描述实现现状。
- `docs/development.md` — 开发者入门与日常开发指引。
- `docs/api.md` — 对外 REST + SSE 接口契约（含 Java 端对接示例）。
- `docs/deployment.md` — 面向运维人员的简明部署文档（含国内网络下的安装渠道建议）。
- `docs/adr/` — 架构决策记录。
- `docs_templates/` — 本地模板库（品类识别与大纲骨架来源）。
- `docs/agents/` — 智能体协作约定（issue 跟踪、triage 标签、领域文档布局）。
