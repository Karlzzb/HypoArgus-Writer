# ADR-0006：章级评审子智能体化与分区式修订说明

状态：已接受（2026-07-22）。
适用范围：第三个业务子智能体 `agents/chapter_reviewer`、分区式修订说明契约（`agents/contracts.py`）、自审裁决项定级配置（各文种 ssot-config `audit_items`）。
部分取代 ADR-0004 决策 1：self_check 折叠职责迁入独立的章级评审；ADR-0004 决策 2、3 不变。
动因：写作与评审耦合在同一子智能体里，评审口径不可独立演进，也无法把评审结论以结构化分区形式交给下游驱动定向修订。

## 1. 章级评审子智能体化

`chapter_reviewer` 与 search_agent、rewriter_loop 同为业务子智能体，登记进 `domain/units.py` 的 `SUBAGENT_UNITS`。
它是任务包 dict 进/出的黑盒异步调用（ADR-0001 约束 3），经 `SubagentAdapter` 包装、`build_graph` 参数注入、stub 可替换。
不设 `CHAPTER_REVIEWER_LLM_MODEL` 覆盖时回落全局 `LLM_MODEL`；外部 LLM 调用经线程信号量限流，与检索子智能体共用同一机制（`agents/concurrency.py`）。

评审为单次调用：确定性 lint → 一次四维 LLM 自审 → 装配修订说明 → 折叠 self_check，评审内部不迭代、不做修一次。
关键步骤经 `SUBAGENT_PROGRESS` 上报（`lint_done`、`llm_call_start`/`llm_call_end`、`audit_done`、`revision_note_done`），载荷只放元数据。
确定性风格校验与风格指南留在 rewriter_loop 包内，评审跨包引用其纯函数（`style_linter.lint` / `audit_items_for` / `CITATION_RULES`），无循环依赖。

## 2. 分区式修订说明契约

`RevisionNotePayload` 分四区：

- 用户指令区（`user_directives`）：revise 时取用户意见原文，逐字保留、零改写；draft 无意见为空串。
- 规则违规区（`rule_violations`）：确定性 lint 违规与四维自审违规折成同形一条，各带位置摘录（`location_excerpt`）、修改指导（`guidance`）与定级（`severity`，error/warn）。
- 冲突提示区（`conflict_hints`）：「规则违规修改与用户指令相抵触」提示；用户指令优先，评审只提示不代改。
- passed 结论：error 级违规为空即过（warn 级不阻断）。

`ReviewResult` = 修订说明 + `self_check`；规则违规区任一条命中 `CITATION_RULES` 则 `citations_ok=False`，交全局终审裁决。
装配为纯函数（`revision_note.assemble_revision_note`）。

## 3. 四维自审经 ssot-config `audit_items` 定级

四维裁决项进各文种 ssot-config `audit_items`（与 lint 同源同机制、两层并集合并，ADR-0005）：派生未标（error）、论证质量（error）、章内连贯（warn）、摘要链一致（warn）。
`AuditItem` 增 `severity` 字段（缺省 error）：定级权威归配置——模型只判违规与冲突、给位置摘录与修改指导，不裁定 severity。
`audit_items` 由 rewriter_loop in-loop 自审与 chapter_reviewer 章级评审共用同一清单，两个消费方口径一致、不重复定义；warn 级不进引用门禁，`citations_ok` 门禁口径不变。
