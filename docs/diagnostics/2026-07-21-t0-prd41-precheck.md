# T0：PRD #41 前置诊断（Langfuse trace 三项分析 + 三个 gate 决策）

对应 issue #43，为 PRD #41（检索漏斗放宽 + 两级评审架构）实施前的必做诊断。
本文档是 spike 数据结论，非代码交付：给出三项可复核的 trace 分析与三个 gate 决策，供 T1–T3a 取用。

## 数据来源与统计口径

分析对象是 2026-07-21 深夜的门控真实链路 E2E 通过跑（即 #40 记录的 818s 那次），经 Langfuse 公开 API 复核。
由于 Send 并行分支根 trace 未传播（#40 附带观察，本次同样复现），一次运行在 Langfuse 里被拆成同一 `thread_id` 下的多条 trace，故按 `thread_id` 归并成「运行」。

| 记号 | thread_id | 首跑 trace | 修订轮 trace | 结构 |
| --- | --- | --- | --- | --- |
| 运行 A（#40 的 818s 跑） | `09ef59239cf14426bad59db8f254dbca` | `43218daf`（653s） | `31d0f101`（156s） | 8 章、7 章检索 |
| 运行 B（同 demo，相邻跑） | `b1b45f85ce594f06bf8090bbc79d58da` | `2fc3e1cf`（705s） | `a3a363a1`（234s） | 8 章、6 章检索 |

运行 A 首跑 653s + 修订 156s + 定稿 ≈ 809s，7 章检索、修订轮 156s，与 #40 的 818s / 7 章 / 152s 一一对应，据此锚定为同一次跑。
分析 2、3 以运行 A 为主、运行 B 为佐证（两跑均为真实 web 检索链路）；分析 1 汇总 A+B 两跑的写作修复调用以增大样本。
检索侧数据取自 `subagent:search_agent` span 的 `search_agent_flow_metrics`（全量诊断，仅进 Langfuse span，不在结束事件摘要里）；写作侧违规取自 `subagent:rewriter_loop` 子生成（`OpenAI-generation`）的 fix 调用入参，其中违规以 `[规则名]` 前缀逐条列出。

复核口径说明：`flow_metrics` 里 `judge_relation_distribution` 记逐（任务行, 引文）的裁决关系；契约 verdict 的 `pass` 严格等于「正向线且被列为 supporting 的引文」，即 `relation==SUPPORT`（见 `src/agents/search_agent/mapping.py:128`、`src/search_agent/evidence_retrieval/output_adapter.py:435`）。SUPPLEMENT、NEUTRAL、REFUTE 一律落 `fail`，被 `extract_chapter_materials` 的 pass-only 过滤挡在写作素材池外。

## 分析一：违规规则分布（供 T3a 首稿提示词前移）

**样本**：运行 A+B 两跑、23 条带违规的 fix 调用入参、合计约 148 条违规实例。
**方法**：从 fix 调用入参正则抽取 `[规则名]` 令牌，与 `style_linter.py` 已注册规则表比对计数。

| 规则 | 实例数 | 层级 | 性质 |
| --- | --- | --- | --- |
| `unknown_material_marker` | 85 | 通用层 | 检索饥饿的症状 |
| `word_count` | 61 | 通用层 | 独立风格约束 |
| `fabricated_quantitative` | 1 | 通用层 | 长尾 |
| `self_audit_unmarked_derived_content` | 1 | 通用层（自审） | 长尾 |
| `oral_blacklist` | 1 | 通用层 | 长尾 |

**结论**：违规高度集中在两条通用层规则，长尾（≤1）可忽略。
首位 `unknown_material_marker`（85）是检索饥饿的下游症状：本跑每章仅 2–5 条 pass 素材（见分析二），写作只能大量插入「待补充」占位标记，故触发大量该违规。把它前移进首稿提示词无助于消除——写作无法引用不存在的素材，根因在检索侧（T1），不在提示词。
真正可前移的是 `word_count`（61）：这是写作可主动满足的确定性约束，前移进首稿提示词能实打实降低首稿一次不过、少触发重写。

**供 T3a 取用**：首稿提示词前移目标锁定 `word_count`（章/节/小节三级字数与均衡约束）；`unknown_material_marker` 不作为提示词前移项，其收敛依赖 T1 检索放宽后素材落库数上升。

## 分析二：候选死于白名单 vs 裁决层（决定白名单是否评分因子化）

**样本**：运行 A+B 两跑、14 个章级检索、约 1629 条原始检索结果、218 个进裁决的候选。
**方法**：汇总各 `search_agent` span 的 web_task 漏斗计数与裁决关系分布。

| 死因环节 | 运行 A | 运行 B | 合计 |
| --- | --- | --- | --- |
| 白名单硬拦（`invalid_url_count`） | 0 | 0 | **0** |
| denylist 硬拦（`denylist_count`） | 0 | 0 | **0** |
| 词法+同域上限软过滤（`lexical_filtered`+`domain_limit`） | 571 | 435 | 1006 |
| 进裁决候选（`judge_input_candidate_count`） | 118 | 100 | 218 |
| → SUPPORT（=pass 落库） | 2 | 3 | **5** |
| → SUPPLEMENT（相关补充，被丢） | 29 | 5 | **34** |
| → NEUTRAL 关系 | 255 | 226 | 481 |
| → REFUTE | 0 | 0 | 0 |

域名白名单在两跑的解析环节硬拦 **0** 条候选。
这与代码一致：`web_whitelist_enabled` 默认 `False`，`.env` 未开启、无内置域名清单（`src/search_agent/evidence_retrieval/config.py:96`）。
运行 A 仅在抓取环节出现 1 次 `WEB_WHITELIST_BLOCKED`，但它是抓取失败回落摘要（`fetch_fallback`），候选仍被保留，未被丢弃。
候选真正大规模消解在两处，都不是白名单：一是词法+同域上限软过滤（1629 原始→约 351 候选），二是**证据裁决层**——218 个进裁决候选里仅 5 个被判 SUPPORT（→pass），约 97.7% 未通过。

**结论**：候选主要死于裁决层，白名单实际未拦任何候选。
#40 「域名白名单再拦一道」的表述在本样本上不成立——白名单是关闭态、贡献为 0。

**Gate 决策——白名单是否评分因子化：不做。**
依据：白名单当前丢弃候选数为 0，把它降为评分因子解决的是一个不存在的瓶颈；检索侧真正的杠杆是查询构造（杠杆①）与裁决产物的利用（杠杆②），与白名单无关。本轮不改白名单形态。

## 分析三：修订轮增量检索 0 落库 vs 落库未注入（决定是否单独开修复票）

**样本**：运行 A 修订轮 trace `31d0f101` 的唯一一次增量检索 span（`8edef100`，约 62s，对应 #40 的「第二章增量检索 ~67s」）。
**方法**：读该 span 的 `flow_metrics` 裁决产物，并对照写作消费链路（`extract_chapter_materials` → `materials_from_segment`）。

增量检索数据：原始 130 条 → 39 候选 → 27 进裁决 → **SUPPORT=0、SUPPLEMENT=0、REFUTE=2、NEUTRAL=62**，`evidence_created_support=0`。

**结论：定性为「0 落库」，非「落库未注入」。**
该轮增量检索裁决产出 0 条 SUPPORT（也 0 条 SUPPLEMENT），故没有任何 pass 素材可落库、更谈不上注入；`extract_chapter_materials` 按 pass-only 过滤后注入 0 条，是正确行为而非缺陷。
#40 观察到的「重写输入未出现新素材、终稿仍只引原有 1 条」由此完全解释：不是消费链路漏注入，而是裁决层没产出可用素材。

**Gate 决策——是否另开增量检索缺陷修复票：不开。**
依据：消费链路（落库→过滤→注入）无缺陷；增量检索 0 落库与首跑同根因（裁决层严格 + 查询构造弱），由 T1 杠杆①（查询聚合论点+假说）与杠杆②（相关补充素材降级落库）一并覆盖，无需独立修复票。
附带建议（不阻塞）：修订轮的增量检索应同样享受杠杆①②，否则会重演「花 60s 产出 0 可用素材」。

## Gate 决策——检索杠杆②（INCONCLUSIVE 降级落库）是否放行

**放行，但收窄口径：仅降级落库 SUPPLEMENT 关系（及非 IRRELEVANT 的近似命中），不做无差别落库。**

依据：被裁决层丢弃的桶里确有可用素材，且可与噪声区分。
运行 A 的 255 条 NEUTRAL 关系按 `neutral_reason_distribution` 拆分为：IRRELEVANT=143、BACKGROUND_ONLY=51、WRONG_METRIC=31、WRONG_ENTITY=14、WRONG_YEAR=8、UNIT_MISMATCH=4、WRONG_REGION=3、MISSING_NUMERIC_VALUE=1。
其中 SUPPLEMENT（两跑 34 条，引文引用校验 `quote_validation_reject_count=0`）是话题相关的补充证据，正是「有观点认为 / 有研究表明」留余地措辞的素材；BACKGROUND_ONLY（51）及 WRONG_METRIC/ENTITY/YEAR 等近似命中是「对题但不精确」的弱佐证。
这些合计约 80+ 条相关但非直接支撑的素材，符合杠杆② 「INCONCLUSIVE 里确有可用素材」的放行条件。

但不能无差别放行：IRRELEVANT（143，占 NEUTRAL 过半）是真无关噪声，若一并落库会用弱引用盖掉「证据薄弱」的警告信号，违背 PRD story 3。
故 T1 实施②时应按裁决产物区分：`relation==SUPPLEMENT`（以及 neutral_reason 属近似命中类）→ 映射 inconclusive 落库、弱佐证措辞；`relation==NEUTRAL` 且 `reason==IRRELEVANT`、`relation==REFUTE` → 仍丢弃。
裁决产物已带 `relation` 与 `neutral_reason`，该区分在数据上可行、无需放宽裁决 prompt。

**下限计数提醒**：本样本每章 pass（SUPPORT）落库 0–3 条，普遍低于缺省下限 3；杠杆② 落库的 inconclusive 素材按 PRD 只单列计数、不计入 pass 下限，下限警告仍应如实暴露薄弱章。

## 三个 gate 决策汇总

| Gate | 判定 | 一句话依据 |
| --- | --- | --- |
| 检索杠杆②（INCONCLUSIVE 降级落库）放行 | **放行（收窄口径）** | 丢弃桶里有 34 条 SUPPLEMENT + 51 条 BACKGROUND_ONLY 等可用弱佐证；但需按 relation/neutral_reason 区分，IRRELEVANT 仍丢 |
| 白名单评分因子化 | **不做** | 白名单实丢候选数为 0（关闭态），非瓶颈 |
| 另开增量检索缺陷修复票 | **不开** | 定性为 0 落库、消费链路无缺陷，根因与首跑同源，由 T1 覆盖 |

## 复现方法

Langfuse 公开 API（`GET /api/public/traces/{id}` 携 observations、`GET /api/public/observations/{id}` 取 span metadata），凭据取自仓库 `.env`。
关键字段：`search_agent_flow_metrics.{web_task,judge_relation_distribution,judge_integrity,neutral_reason_distribution}`（检索侧）、`subagent:rewriter_loop` 子生成入参的 `[规则名]` 令牌（写作侧）。
锚定运行按 `thread_id` 归并 trace，以首跑+修订耗时与检索章数与 #40 的 818s/7 章/152s 对齐。
