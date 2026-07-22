# ADR-0007：revise 路径评审前置与修订说明统一驱动

状态：已接受（2026-07-22）。
适用范围：writing_orchestrator 的修订与终审回退路径（`src/nodes/writing_orchestrator.py`）、修订相关契约（`src/agents/contracts.py`）、修后终态自检纯函数（`src/nodes/chapter_write_loop.py`）。
承接 ADR-0006：分区式修订说明从写作首写循环推广为全部修订链路的唯一驱动结构；ADR-0004 决策 2「revise 即修、不二次调用」的收敛边界在此进一步固化。
动因：人工修订轮存在「按指令重写 → 终态校验 → 按残余违规二次重写」的叠加，成本与时延翻倍且两次改写口径不一；终审打回仍以拼接的一句话字符串指令驱动改写，与 ADR-0006 的分区式修订说明并存两套指令形态，契约层旧字段（`RevisionDirectivePayload` / `revision_directives`）成为需要收口的兼容负担。

## 1. revise 路径评审前置，终态只记录不复审

人工修订指令先经 `chapter_reviewer`（mode=revise）装配分区式修订说明，再调 `rewriter_loop`（mode=revise）恰一次改写。
评审在改写之前发生：用户意见原文经任务包 `user_feedback` 逐字进入修订说明的用户指令区（零改写，ADR-0006 分区语义不变），现有正文的规则违规与冲突提示由评审一并折入对应分区。
改写完成后的终态以 `relint_self_check` 记录：纯函数、零 LLM，按与 rewriter / reviewer 同源的确定性 lint 折出修后终态自检，不复审、不触发二次重写。
由此消除「重写→校验→二次重写」叠加：修订轮恰一次 LLM 改写调用，终态仍存的违规如实折入 `self_check` 交全局终审在重试预算内裁决，收敛边界与 ADR-0003、ADR-0004 一致。

## 2. 终审打回统一为修订说明驱动

终审报告即评审结论，打回时不再经 chapter_reviewer，也不再拼接一句话字符串指令。
每条 `CitationIssue` 直接折成一条 error 级 `RuleViolationEntry`：`rule` 取 `citation.<kind>`、`guidance` 取 `detail`、位置摘录为空串。
组装为纯函数（`_report_revision_note`）：无用户指令、无冲突提示，存在 error 级违规故 `passed=False`，以 `RevisionNotePayload` 驱动 `rewriter_loop`（mode=revise）恰一次改写。

## 3. 收口删除契约层旧修订指令字段

删除 `agents/contracts.py` 中的 `RevisionDirectivePayload` 与 `RewriteTask.revision_directives`。
三条修订链路——写作首写循环（写→评→重写）、终审打回、人工修订——统一消费 `RewriteTask.revision_note`（`RevisionNotePayload`），rewriter_loop 不再感知第二套指令形态。
State 层的 `RevisionDirective`（`domain/state.py`）保留：它是人工评审产物的领域事实，继续驱动修订模式的目标章选取与 `evidence_augmented` 增量检索判定；其指令文本经评审前置（决策 1）折入修订说明，不再直接进入写作任务包。

## 后果

- 修订轮的 LLM 写作调用从至多两次降为恰一次，成本与时延减半，且改写口径统一由分区式修订说明给出。
- 修订指令形态归一后，rewriter_loop 保持纯写作职责（ADR-0006），提示词装配只需处理一种修订输入。
- 终审打回绕过 chapter_reviewer 属有意取舍：终审报告本身就是结构化评审结论，再过一次评审是重复 LLM 花费且不产生新信息。
- 中断续跑语义不变：修订与打回仍按超步逐章落 checkpoint，崩溃重跑只损失进行中的分支（ADR-0001 约束 1）。
