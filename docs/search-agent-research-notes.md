# 检索子智能体内部流程与研究点

检索子智能体的分层、判断逻辑与待研究的开放问题。
实现事实源为 `src/agents/search_agent/` 与 `src/search_agent/evidence_retrieval/`；本篇只做研究梳理，不构成决策，决策落 ADR 后再改代码。

## 1. 分层

- **适配层** `src/agents/search_agent/`（`agent.py`、`mapping.py`、`runtime.py`）：黑盒 dict 进/出，不做 LangGraph 子图化。
- **引擎** `src/search_agent/evidence_retrieval/`：真实检索编排，主编排图在 `flows/parallel_sources_flow.py`。

适配层只翻译契约与发诊断摘要，引擎承担全部检索与判断。

## 2. 检索依据：从任务包到查询

任务包字段：`chapter_id`、`points[]`、`hypotheses[]`（每条含 `text` 与 `refute_condition`）、`genre`、`existing_materials_digest`。

`mapping.engine_payload_from_task`（`mapping.py:47`）把每条假说映射为一个正向检索项（`claim`，目标文本为假说本文）；`refute_condition` 非空者另映射一个反向检索项（`oppose`，目标文本为反驳条件），落实可证伪设计。
`genre` 进论证边界字段作检索范围提示；论点经 `argument_path` 给引擎论证层级上下文。

引擎 `prepare_tasks` 阶段用 `atomize_claim`（`claim_logic.py:77`）把检索项 `target_text` 确定性拆成原子主张，抽取 `subject`/`metric`/`time_scope`/`value+unit+operator`/`polarity`/`logic_operator`。
反向项另经 `normalize_reverse_hypothesis`（`claim_logic.py:131`）把"是否/有没有"问句转成中性可检索陈述句并剥除比较词。

查询构造（query_build 阶段）从原子主张字段派生多个短变体，绝不拼接完整 `target_text`：
- Web 查询 `build_web_query_variants`（`parallel_sources_flow.py:245`）：`time+subject+metric` / `time+subject+numbers` / `subject+metric+报告` / `+统计` / `+官方数据`，取 3–4 个、各 ≤120 字。
- KB 查询 `build_kb_query_variants`（`retrieval_queries.py:15`）：`full`/`neutral`/`subject_metric_year`/`numeric_gap`，单原子主张且主语年份数值已在原句时收敛为单查询避免冗余扇出。

三通道并行（`parallel_sources_flow.py:963`）：
`validate → prepare_tasks → query_build → web_search → web_filter → web_fetch → web_bm25 → selected_kb_retrieve → public_kb_retrieve → structured_match → structured_query → candidate_merge → batch_judge → verification → finalize`。
`candidate_merge`（`channels.py`）按 `(task_id, source_type, fingerprint)` 去重；槽位未覆盖时 `gap_retrieval` 补检一轮。

## 3. 判断逻辑：三段门控

### 3.1 证据裁判（LLM 批量判关系）

`StructuredLLMBatchEvidenceJudge.judge_many`（`evidence_judge.py:287`）把一批 task×candidate 合并为一次 LLM 调用，prompt 含段落上下文、`boundary`、既有论据、`required_slots`、原子主张清单、候选内容（BM25 截窗，标 `<evidence>` 并声明不可信不执行）。
LLM 对每候选×每原子主张返回 `SUPPORT/REFUTE/SUPPLEMENT/NEUTRAL` + `confidence` + `directness` + `quoted_spans` + `covered/missing_slots`。

解析层 `parse_batch_judge_response`（`evidence_judge.py:1376`）健壮：Markdown 代码块、平衡 JSON、对象或数组容器、字段别名、`neutral_results` 紧凑表、解析失败格式修复重试。
契约强制每个期望候选恰好一条结果，缺失项报错，**禁止自动补齐为 NEUTRAL**。

防捏造引文 `validate_judgement`（`evidence_judge.py:93`）：非 NEUTRAL 必须带 `quoted_spans`，且引文须真实存在于候选内容（NFKC 归一化，容忍 PDF 换行/千分位），否则强制降级 NEUTRAL、`confidence≤0.25`。

原子主张聚合到候选级关系（`evidence_judge.py:1306`）：有 REFUTE → REFUTE；有 SUPPORT → SUPPORT；有 SUPPLEMENT → SUPPLEMENT；否则 NEUTRAL。

### 3.2 引文门控（决定候选是否进公开引文/素材）

`output_adapter._citation`（`output_adapter.py:216`）按序卡多道门，任一不通过 `return None`：
1. `relation == NEUTRAL` 丢弃（`:275`）。
2. 内容须为完整事实句。
3. `scope_mismatch_reasons` 不含阻断项。
4. `reason` 不含"无法直接支持/确认/反驳/否定"。
5. SUPPORT/REFUTE：须有合法 `supported/refuted_claim_ids`、`scope_compatible=True`、`confidence ≥ 0.60`、`directness ≥ 0.60`、每 claim 完整事实句。
6. SUPPLEMENT：`matched_claim_id` 合法、`confidence ≥ 0.40`。

通过者产 `CitationRecord(relation=SUPPORT/REFUTE/SUPPLEMENT)`；NEUTRAL 候选根本不进任何引文清单。

### 3.3 章级裁决折算

充分性（`config.py:159`、`verification.py:47`）：`direct_evidence≥1`、`effective_evidence≥2`、`independent_document≥2`、`independent_source≥1`、`claim_coverage≥0.65`、`final_evidence_score≥0.70`、`noise_ratio≤0.60`、非仅 snippet、无 `missing_slots`。
verdict：双方都过冲突阈 → CONFLICT；sufficient 且 `support-refute ≥ 0.25` → SUPPORTED（反向 REFUTED）；权威结构化覆盖；否则 INCONCLUSIVE。
原子主张逻辑 `apply_claim_logic`（AND 短路：一分支 REFUTED 即整句 REFUTED）优先级高于质量裁决；`missing_slots` 把 SUPPORTED/REFUTED 降级 INCONCLUSIVE；内部与原子逻辑不一致则降 INCONCLUSIVE。

引文 ID 组装（`output_adapter.py:435-567`）：
`supporting = [cid for SUPPORT]`、`supplementary = [cid for SUPPLEMENT]`、`refuting = [cid for REFUTE]`，`citation_ids` 为三者并集。

### 3.4 适配层折算成素材

`mapping.search_result_from_engine_output`（`mapping.py:114`）遍历 `results[].citation_ids`：
正向线 + `cid ∈ supporting` → **pass**；正向线 + `cid ∈ supplementary` → **inconclusive**；其余 → **fail**。
同 (假说,引文) 多线取强者（pass > inconclusive > fail）。
薄弱章暴露：`pass_count < SEARCH_AGENT_MIN_PASS_PER_CHAPTER`（缺省 3）发 `weak_chapter_warning`，不阻断不补检（`agent.py:194`）。

## 4. 研究点（待定性、未动代码）

以下为 issue #52 残留与一般性开放问题，按可测性排序。

### R1：0 pass 根因定性（最该先做，确定性可测）

探针实证召回 2 条全 fail、0 pass。
代码已证伪键名错位假设：引擎 `output_adapter.py:565-567` 填的 `supporting/supplementary/citation_ids` 与 `mapping.py:141-156` 读的三键完全一致，契约无错位。
因此 0 pass 必落召回质量或裁判倾向：低权威新闻页被判 NEUTRAL（门控第 1 道丢弃）或 REFUTE/SUPPLEMENT（即便过门也只产 fail/inconclusive）。
**下一步**：扩 `scripts/probe_search_agent.py` dump 原始 `decision` 三键与每候选的 `relation`/`confidence`/`directness`/`scope_compatible`/`quoted_spans`，区分"判了 SUPPORT 但被门控卡掉（阈值问题）"与"根本没判 SUPPORT（召回质量问题）"。

### R2：裁判阈值是否偏严

`public_citation_min_confidence=0.60`、`public_citation_min_directness=0.60`、`min_final_evidence_score=0.70`、`verdict_margin=0.25`。
若 R1 显示有候选判 SUPPORT 但被门控卡，需定量评估这些阈值对低权威来源是否过严，再决定是否分级调整。
阈值改动影响全局，须配真实链路回归（FakeLLM 测不出，见 [[real-e2e-regression-check]]）。

### R3：召回质量与查询有效性

Web 召回到 `auto.sina.cn`/`m.sohu.com` 等低权威新闻页。
待研究查询变体（`build_web_query_variants` 的 `+报告`/`+统计`/`+官方数据`）对权威来源的命中率，以及是否需引入权威域名偏好（`config.preferred_domains` 已存在但未量化效果）。

### R4：裁判契约的 NEUTRAL 丢弃边界

门控第 1 道把 NEUTRAL 候选直接丢弃，不进任何引文清单也不进素材。
这意味着裁判判 NEUTRAL 的候选在写作侧等同空池，与 fail 同效。
待研究是否需要把 NEUTRAL 也以 fail 形态留痕进审计池，供写作侧"无可用素材"的显式判断而非隐式空池。

### R5：单次 LLM 调用延迟与并行收益

framework 阶段单次大纲+假说调用约 67s 串行。
待研究拆成"大纲→各章假说扇出"的前移并行收益上限（耦合弱，收益有上限）。

### R6：篇级终审吃掉的时长

真实 E2E 总 17.5min，疑 `document_reviewer` 反复全篇评审（48 次 retry 事件）。
待从 trace 抽每次起止求和量化，与问题一基本独立。
