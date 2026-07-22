# ADR-0006：章级评审子智能体化与分区式修订说明

状态：已接受（2026-07-22）。
适用范围：新增第三个业务子智能体 `agents/chapter_reviewer`，分区式修订说明契约（`agents/contracts.py`），以及自审裁决项定级配置（各文种风格指南 ssot-config `audit_items`）。
与既有 ADR 的关系：遵守 ADR-0001 四条硬约束；沿用 ADR-0005 的「自审裁决项进各文种 ssot-config、两层合并」机制；**部分取代 ADR-0004**（详见「后果」）。

## 背景

ADR-0004 把 self_check 折叠职责压在 rewriter_loop 编排体内：写作与评审耦合在同一子智能体里，评审口径不可独立演进，也无法把评审结论以结构化、分区的形式交给下游驱动定向修订。
修订指令（ADR-0004 的 `revision_directives`）是扁平的「类型 + 指令」清单，既不保留用户意见原文，也不区分规则违规的定级，更没有「用户指令与规则冲突时以谁为准」的显式表达。

本项目需要一个独立的章级评审环节，产出一份**分区式修订说明**作为单一事实源：用户意见逐字可溯、规则违规逐条可定位可定级、冲突处用户指令优先、结论清晰（error 级为空即过）。

## 决策

### 1. 章级评审子智能体化

新增 `chapter_reviewer`，与 search_agent、rewriter_loop 同为业务子智能体，登记进 `domain/units.py` 的 `SUBAGENT_UNITS`。
它是任务包 dict 进/出的黑盒异步调用（ADR-0001 约束 3：非子图边界），经 `SubagentAdapter` 包装、`build_graph` 参数注入、stub 可替换。
模型保持 plus：不设 `CHAPTER_REVIEWER_LLM_MODEL` 覆盖时回落全局 `LLM_MODEL`。
外部 LLM 调用经线程信号量限流，与检索子智能体共用同一机制（抽入 `agents/concurrency.py`，单一事实源）。

评审为**单次调用**（single-shot）：确定性 lint（纯函数、零成本）→ 一次四维 LLM 自审 → 装配修订说明 → 折叠 self_check，评审内部不迭代、不做修一次。

关键步骤经 `SUBAGENT_PROGRESS` 事件对外上报（ADR-0001 约束 2），载荷只放元数据：lint 完成（`lint_done`）、自审调用（`llm_call_start`/`llm_call_end`，call=audit）、自审结论（`audit_done`）、修订说明生成（`revision_note_done`）。

确定性风格校验与风格指南留在 rewriter_loop 包内；评审**跨包引用其纯函数**（`style_linter.lint` / `audit_items_for` / `CITATION_RULES`），无循环依赖。

### 2. 分区式修订说明契约

`RevisionNotePayload` 分四区：

- 用户指令区（`user_directives`）：revise 时取用户意见原文，**逐字保留、零改写**；draft 无意见为空串。
- 规则违规区（`rule_violations`）：确定性 lint 违规与四维自审违规折成同形一条，各带位置摘录（`location_excerpt`）、修改指导（`guidance`）与定级（`severity`，error/warn）。
- 冲突提示区（`conflict_hints`）：模型给出的「规则违规修改与用户指令相抵触」提示；用户指令优先，评审只提示不代改。
- passed 结论：error 级违规为空即过（warn 级不阻断）。

评审结果 `ReviewResult` = 修订说明 + `self_check`；`self_check` 按引用类规则折叠——规则违规区任一条命中 `CITATION_RULES` 则 `citations_ok=False`，交全局终审（citation_validator）裁决。
装配为纯函数（`revision_note.assemble_revision_note`），单测可独立断言逐字保留与定级口径。

### 3. 四维自审经 ssot-config `audit_items` 定级

四维裁决项进各文种风格指南的 ssot-config `audit_items`（与 lint 同源同机制、两层并集合并，ADR-0005）：派生未标（error）、论证质量（弱素材写成断言，error）、章内连贯（warn）、摘要链一致（warn）。
`AuditItem` 增 `severity` 字段（缺省 error）：各文种经逐条 `severity` 声明实现「定级」，文种层追加自身裁决项实现文种粒度「开关」。
定级权威归配置——模型只判违规与冲突、给位置摘录与修改指导，不裁定 severity，避免模型漂移影响门禁。

`audit_items` 由 rewriter_loop in-loop 自审与 chapter_reviewer 章级评审**共用同一清单**：自审裁决项是「本章正文是否违规」的单一事实源，两个消费方口径一致、不重复定义。

## 后果

- ADR-0004 决策 1（self_check 折叠在 rewriter 编排体内）被本 ADR 部分取代：折叠职责迁入独立的章级评审，rewriter 的 in-loop self_check 暂留、待 T3 接入评审后再收束；ADR-0004 决策 2（revise/fix 合并）与决策 3（引文门禁前移写作提示词）不变。
- 新增三维裁决项进通用层 `audit_items` 后，rewriter 的 in-loop 自审也会评估这三维（共用清单的直接后果）：warn 级不进引用门禁、error 级中仅「派生未标」属引用类，故 `citations_ok` 门禁口径不变；受影响的裁决项计数类测试按 SSoT 变更同步更新。
- 写作任务包 `RewriteTask` 增可选 `revision_note` 字段，与旧 `revision_directives` **并存**（expand）：本期只落契约，rewriter 消费与旧字段删除留 T3/T3b。
- 章级 checkpoint 由编排层负责（ADR-0001 约束 1）：评审接入写作自环时按超步落 state，本 ADR 只落子智能体本体，不改自环时序。
