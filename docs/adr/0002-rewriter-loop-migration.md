# ADR-0002：rewriter_loop 真实现迁移的架构决策

状态：已接受（2026-07-19）；「修后不复检」已被 ADR-0004 取代，tier/doc_type 环境变量已被 ADR-0005 取代，此处只保留仍然有效的决策。
适用范围：`agents/rewriter_loop/` 真实现子包及其在主图与服务层的装配。

## 契约零扩展

任务包与结果 dict 逐字遵守 `agents/contracts.py` 既有契约，零扩展。
参考文献职责上收至书目渲染层（学院风格走 gbt7714 渲染器，注册表可扩展），正文不夹带章内参考文献列表。
校验违规明细折叠进 `self_check.issues`（形如 `[规则名] 说明`）；是否修订过走进度事件（`revise_triggered`），不进结果 dict。

## 引用角标统一

引用角标统一为本项目单方括号 `[素材id]` 语义，可并列叠加、同素材复用同 id。
角标解析复用 `domain.citation_reconciler.MARKER_PATTERN`，不另定义正则。
校验规则按主项目语义建模：角标 id 必须在素材池内（`unknown_material_marker`）、照抄素材原文须挂角标（`unmarked_derived_content`）等，结论进 `self_check`。

## LLM 栈归一

不用 langchain-openai：真实适配器 `LlmWriterClient`（`llm_adapter.py`）的底座是本项目 LLM 协议实例，按单元名 rewriter_loop 经 LLM 工厂构造（`REWRITER_LOOP_LLM_*` 前缀回落全局）。
结构化输出走 prompt 要求 JSON + 文本解析（`llm.llm_json.parse_json`），这是「单一技术栈」的刻意代价，由退化重试兜住解析失败：

- draft/revise：异常、解析失败、结构非法或空正文均视为退化，重试至上限 3 次；拿到过合法信封则返回最后一次诚实结果并标 `degraded`；从未拿到信封但抛过异常则重抛最后一个异常。
- audit：空裁决（`issues: []`）是合法非退化结果、不重试；重试耗尽降级为空裁决（`degraded=True`），自审永不阻断主链。

## 包内注入点与落位

包内 LLM 注入点为 `WriterLlmClient` 协议（draft/revise/audit 三方法），配确定性 `FakeWriterLlmClient` 驱动全部编排与校验测试，零网络零成本。
子包构成：编排（`writer.py`）、注入点（`writer_client.py`）、适配器（`llm_adapter.py`）、风格校验器（`style_linter.py`）、随包风格指南（不可配，ADR-0005 后为 `style_guides/` 目录）。
打桩实现同包共存（`stub.py`），可显式注入；`build_graph` 与 `create_app` 缺省使用真实现（demo 空转也走真链路，仅最底层模型是 FakeLLM）。
真网络验证不进 CI：走 `scripts/rewriter_debug.py --real` 与 `scripts/demo.py --real`。

## 主框架开口子严格限于 ADR-0001

主框架侧只做 ADR-0001 明文要求的三处开口子：章级 checkpoint 自环、`subagent_start`/`subagent_end` 载荷携带 chapter_id 与调用模式、`SUBAGENT_PROGRESS` 五个发射点（`llm_call_start`、`llm_call_end`、`lint_done`、`audit_done`、`revise_triggered`，只放元数据不放正文全文）。
其余分毫不动：图结构、`Subagent` 协议形状、运行单元名册、State schema、事件信封类型枚举。
