# ADR-0008：篇级终审扩维与两级评审视野切分

状态：已接受（2026-07-22）。
适用范围：篇级终审主节点（`src/nodes/document_reviewer.py`，原 citation_validator 改名）、图状态与路由（`src/domain/state.py`、`src/graph.py`）、篇级全文装配段（`src/assembly/context_assembler.py`、`src/assembly/assembler_config.py`）、人工中断点载荷（`src/nodes/human_review_gate.py`）、终审打回改写驱动（`src/nodes/writing_orchestrator.py`）。
承接 ADR-0006（章级评审子智能体化）与 ADR-0007（打回统一为修订说明驱动）。
动因：终审节点原本只做引文核查，跨章硬事实冲突、章间衔接、口径统一、跨章重复等「必须看全篇才能裁」的维度无人负责；章级评审落地后，需要一次性划清两级评审的职责边界，避免维度重叠或漏裁。

## 1. 篇级终审扩维与维度准入原则

citation_validator 改名为 document_reviewer，定位从「引文终审门禁」升级为「篇级终审门禁」。
既有引用四步原样保留：纯程序对账（reconcile）→ 结构完整性纯程序校验（章节编号连续唯一 + 大纲章节缺稿 chapter_missing）→ 合并单章自检结果 → 逐章 LLM 引文语义核查。
在此之上新增篇级 LLM 评审：一次调用、始终全量，评四个维度——跨章硬事实冲突（fact_conflict）、章间衔接（transition）、口径统一（consistency）、跨章重复（duplication）。
维度准入原则：篇级终审只收「必须看全篇才能裁」的维度；凡单章视野内可判定的质量问题，一律归章级评审，不得进入篇级清单。

## 2. 两级评审视野切分

| 层级 | 执行体 | 维度 | 视野 |
| --- | --- | --- | --- |
| 章级评审 | chapter_reviewer 子智能体（ADR-0006） | 派生未标、论证质量、章内连贯、摘要链一致 | 只裁单章内部，不看全篇 |
| 篇级终审 | document_reviewer 主节点 | 引用四步、结构完整性、跨章硬事实冲突、章间衔接、口径统一、跨章重复 | 只裁必须看全篇才能判的维度 |

两级互补不重叠：章级评审在写→评→重写循环内消化章内问题，篇级终审在全部成稿后做全篇裁决。
引用四步中 revised_chapter_ids 非空时增量核查；结构完整性与篇级评审是全文属性，始终全量。

## 3. 严重级裁决权在代码，warn 不打回

模型只报维度、涉及章节与线索，error/warn 归属由代码中的维度表固定，模型输出不参与定级。
error 只收确定性硬伤：引用对账不符类问题、跨章硬事实冲突（fact_conflict）、结构完整性问题（chapter_missing / numbering_broken），进 issues 与 failed_chapter_ids 触发定向打回。
章间衔接、口径统一、跨章重复三个语义维度一律 warn：写入新 state 字段 `review_warnings`，随人工中断点载荷（payload 键 `review_warnings`）呈现给人工，不打回、不计入重试。
warn 不打回是刻意的雪崩防护：篇级语义提示往往牵连多章且判定模糊，若允许打回极易触发多章连锁重写，成本失控且未必收敛；交人工裁量是更稳的收敛点。
幻觉防护同样在代码侧：未知维度直接丢弃，涉及章节 id 不在大纲内的先剔除、剔空后整条丢弃。

## 4. 打回统一走修订说明结构

终审失败在重试预算内（`DOCUMENT_REVIEW_MAX_RETRIES`，缺省 2）定向回退 writing_orchestrator，超限不再回退、携未决警告交人工。
打回改写沿用 ADR-0007 的统一驱动：每条 error 级问题折成一条 error 级规则违规，`rule` 取 `document_review.<kind>` 前缀，组装为 `RevisionNotePayload` 驱动 rewriter_loop（mode=revise）恰一次改写，目标章由 failed_chapter_ids 明确给出。
不新增第二套打回指令形态。

## 5. 全篇装配复用链式压缩策略

篇级评审的全文输入由装配段 `document_text` 提供：全部章节按大纲顺序拼为「## 章节id 标题 + 正文」。
超过阈值（`ASSEMBLER_DOCUMENT_TEXT_MAX_CHARS`，缺省 30000）时复用既有摘要链压缩骨架：保末章原文、更早章折为首句摘要、仍超限则丢弃最早章并加省略注记。
不为篇级评审另造压缩机制，保证单次 LLM 调用输入不越预算。

## 6. 保留旧序列化名的兼容决策

为稳定历史 checkpoint 与对外契约，以下旧名一律保留、不随节点改名：
`CitationIssue` / `CitationReport` 模型名，state 键 `citation_report` / `citation_retry_count` / `citation_warnings`，状态机值 `WorkflowStatus.CITATION_CHECKING`，事件字段 `loop_iteration` 的 `citation_retry`，中断载荷键 `citation_warnings`。
这些名字如今承载篇级终审全部 error 级问题（不止引文），语义扩大而名称不动，属有意取舍：改名收益是可读性，代价是旧 checkpoint 反序列化断裂与对外契约破坏，不划算。

## 7. 环境变量改名清单

| 旧名 | 新名 | 缺省 |
| --- | --- | --- |
| `CITATION_MAX_RETRIES` | `DOCUMENT_REVIEW_MAX_RETRIES` | 2 |
| `CITATION_MAX_CONCURRENT_CHAPTERS` | `DOCUMENT_REVIEW_MAX_CONCURRENT_CHAPTERS` | 4 |
| `CITATION_VALIDATOR_LLM_*` 三项 | `DOCUMENT_REVIEWER_LLM_*` 三项 | 回落全局 |

新增 `ASSEMBLER_DOCUMENT_TEXT_MAX_CHARS`（缺省 30000），归上下文装配阈值组。
环境变量是部署面配置而非序列化契约，随节点改名一步到位，不留旧名别名。

## 后果

- 跨章维度首次有了明确责任方，且准入原则（必须看全篇才能裁）使两级评审边界可长期维持，不随维度增减而糊化。
- 篇级评审仅增加一次全量 LLM 调用，语义维度不打回，最坏情况下的重写成本与改名前持平。
- warn 提示每轮如实呈人工，语义模糊问题的裁量权在人不在机，系统收敛性不受模型判定波动影响。
- 旧序列化名与新节点名并存造成一定阅读负担，由模型 docstring 与本 ADR 显式说明兜底。
