# ADR-0002：rewriter_loop 真实现迁移的架构决策

状态：已接受（2026-07-19）。
适用范围：`agents/rewriter_loop/` 真实现子包、其在主图与服务层的装配，以及围绕它的测试与调测设施。
与 ADR-0001 的关系：本次迁移的主框架改动就是 `docs/adr/0001-subagent-real-impl-constraints.md` 预告的开口子，验收以其四条约束为准；本 ADR 只记录迁移本身的决策，不复述该四条约束。
与 ADR-0004 的关系：本 ADR 记录的「修后不复检（v1）」与「revise 后可再触发修一次」已被 ADR-0004 取代（修后复检 v2、revise/fix 合并）；其余迁移决策不变。

## 背景

真实的章节写作能力此前在独立仓库 HypoArgus-RewriteLoop 中开发完成（draft → 确定性风格校验 → LLM 自审 → 违规修一次），但技术栈与本项目规范不一致：langchain-openai、双方括号引用标记、章内 reference_list。
本仓库的 rewriter_loop 此前只有打桩实现，产出占位正文。
本次迁移把源仓库核心能力搬入本仓库并改造为完全符合本项目框架规范的真实现（GitHub issue #8）。

## 决策

### 迁移方向与不迁资产

迁移是单向的代码搬入与改造，不是联通两个项目：源仓库 HypoArgus-RewriteLoop 一个字不动，迁移完成后不再是运行时依赖。
以下资产一律不迁：promptfoo 评测（真 LLM 质量评测体系后续单独立项决策）、源测试用例与 fixtures（测试按本项目规范全新编写）、语料、tracing 封装（Langfuse 观测在本项目既有 LLM 层与子智能体 span 包装中自动生效）、过时文档（源仓库引用已删 API 的集成文档不作为实现依据）。

### 契约零扩展

任务包与结果 dict 逐字遵守本项目 PRD 现契约（mode/chapter_spec/materials/prev_chapter_summary/revision_directives/current_text → chapter_text/chapter_summary/self_check），零扩展。
源项目的 tier（本科/高职）、doc_type、风格指南路径不进任务包：

- tier 与 doc_type 用环境变量 `REWRITER_LOOP_TIER`（只接受本科|高职，缺省本科）与 `REWRITER_LOOP_DOC_TYPE`（自由文本，缺省人才培养方案）配置，工厂 `make_rewriter_loop` 内读取一次（`load_writer_settings`）。
- 风格指南随包携带不可配（`agents/rewriter_loop/style_guide.md`，代码级单一事实源）；测试可显式传路径覆盖，运行时不可配。

源项目的 reference_list、lint_report、revised 三个输出不进契约，各自去向：

- 参考文献职责上收至书目渲染层（学院风格走 gbt7714 渲染器，注册表可扩展），正文不夹带章内参考文献列表。
- 校验违规明细折叠进 `self_check.issues`（形如 `[规则名] 说明`）。
- 是否修订过走进度事件（`revise_triggered`），不进结果 dict。

### 引用角标统一

引用角标统一为本项目单方括号 `[素材id]` 语义，可并列叠加、同素材复用同 id。
角标解析复用 `domain.citation_reconciler.MARKER_PATTERN`，不另定义正则。
源项目围绕双方括号与章内参考文献的校验规则改写为主项目语义：角标 id 必须在素材池内（`unknown_material_marker`）、照抄素材原文须挂角标（`unmarked_derived_content`）等，结论进 `self_check`。

### LLM 栈归一

弃 langchain-openai：真实适配器 `LlmWriterClient`（`llm_adapter.py`）的底座是本项目 LLM 协议实例，按单元名 rewriter_loop 经 LLM 工厂构造（`REWRITER_LOOP_LLM_*` 前缀回落全局）。
结构化输出从 function_calling 改为 prompt 要求 JSON + 文本解析（`llm.llm_json.parse_json`）。
这是「单一技术栈」的刻意代价，由保留的退化重试模型兜住解析失败：

- draft/revise：异常、解析失败、结构非法或空正文均视为退化，重试至上限 3 次；拿到过合法信封（哪怕正文为空）则返回最后一次诚实结果并标 `degraded`；从未拿到信封但抛过异常则重抛最后一个异常。
- audit：空裁决（`issues: []`）是合法非退化结果、不重试；重试耗尽降级为空裁决（`degraded=True`），自审永不阻断主链。

### 包内缝与「修一次」链路

包内 LLM 缝 `WriterLlmClient` 协议（`writer_client.py`）扩为 draft/revise/audit 三方法，配确定性 `FakeWriterLlmClient`（按顺序消费的脚本化信封 + 调用记录），驱动全部编排与校验测试零网络零成本。
revise 是新写的独立 prompt 路径（同一上下文块 + 现有正文 + 定向修订指令，未被指令覆盖的内容与角标保持原样），与 draft 共享「校验-自审-修一次」链路。
「违规修一次、修后不复检」的 v1 设计原样保留：`self_check` 折叠的是修前质检结论，修订产物是否真正规避了违规由全局终审兜底，双层校验分工不变。

### 落位与装配

新建 `agents/rewriter_loop/` 子包：编排（`writer.py`）、LLM 缝（`writer_client.py`）、真实适配器（`llm_adapter.py`）、风格校验器（`style_linter.py`）、风格指南文件（`style_guide.md`）。
打桩实现保留同包共存（`stub.py`），仍可显式注入，供既有测试与空转装配使用。
真工厂签名为 `make_rewriter_loop(llm_factory, event_hook)`。
总装后真实现是缺省：`build_graph` 与 `create_app` 在未显式注入 rewriter_loop 时使用 `make_rewriter_loop` 构造真实现（demo 空转也走真链路，仅最底层模型是 FakeLLM）。

### 主框架开口子严格限于 ADR-0001

主框架侧只做 ADR-0001 明文要求的三处开口子：

- 章级 checkpoint：写作编排节点图内自环、每超步只写一章，修订分支与终审回退同理（约束 1）。
- 事件载荷扩充：`subagent_start`/`subagent_end` 载荷在单元名之外携带 chapter_id 与调用模式（约束 2）。
- 内部进度：`SUBAGENT_PROGRESS` 事件的五个发射点——`llm_call_start`、`llm_call_end`、`lint_done`、`audit_done`、`revise_triggered`，载荷统一带 unit、chapter_id、mode、step 与环节要点（尝试轮次、正文长度、违规数、是否降级），只放元数据不放正文全文。

其余一律分毫不动：图结构、`Subagent` 协议形状、运行单元名册、State schema、事件信封类型枚举。

### 测试策略：三层缝

只测外部行为不测实现细节，接缝自高向低三层：

- 图级端到端：`tests/e2e/test_graph_e2e.py` 的中断恢复用例走真实现完整链路（真编排、真校验器、真 JSON 解析路径），仅最底层用 FakeLLM 键控应答；覆盖逐章 checkpoint、已完成章零重跑、事件成对与父子链、产物与不中断路径等价。
- 包内缝：`WriterLlmClient` 配 Fake 驱动编排测试（双模式、违规修一次、self_check 折叠）；风格校验器规则单测；adapter 单测（JSON 解析、退化重试、audit 降级）用 FakeLLM 预置文本应答。
- 契约缝：既有子智能体契约测试保留并继续通过，真实现补充同形态断言。

真网络验证不进 CI：走调测脚本 `scripts/rewriter_debug.py --real` 与 `scripts/demo.py --real`；后者每次运行落盘构建过程档案供人工审核。

## 后果

- rewriter_loop 交付真实中文教务公文章节正文，打桩降级为可显式注入的测试替身。
- 章节质量有底线保障（确定性校验 + LLM 自审 + 修一次），但修后不复检意味着修订产物的最终合规由下游终审裁决——`citations_ok=False` 语义为「修前检出过引用类违规、已修一次但未复核」。
- JSON-in-text 解析比 function_calling 更依赖模型遵从性，模型抖动由退化重试与诚实降级吸收，单点故障不拖垮整篇写作。
- prompt 与风格规则的调优不再依赖主图全流程，调测脚本支持按模式与按环节分步执行。
- search_agent 真实现（含 Material 学术要素充实）、子智能体子图化评估、修订循环收敛式多轮迭代均不在本次范围内，后续另行决策。
