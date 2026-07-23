# 构建过程档案

## 运行元信息

- 运行模式：空转（确定性假 LLM）
- thread_id：`4a70180cc6bc476d88101f79e3bbff2f`
- 开始时间：2026-07-23 20:18:24
- 结束时间：2026-07-23 20:19:03
- 模型单元配置摘要（来自 llm_config_used 事件，已脱敏，不含任何密钥）：
  - `chapter_drafter`：`{"unit":"chapter_drafter"}`
  - `document_reviewer`：`{"unit":"document_reviewer","model":"fake-llm","base_url":"fake://"}`
  - `framework_orchestrator`：`{"unit":"framework_orchestrator","model":"fake-llm","base_url":"fake://"}`
  - `reference_orchestrator`：`{"unit":"reference_orchestrator"}`

## 完整事件流

### graph_event 可视化通道

逐条按到达顺序记录：事件类型、单元、关键载荷字段与父子链
（id/parent 为 event_id 前 8 位，parent 指向父事件，可据此审计执行拓扑）。

  1. `2026-07-23T12:18:27.455970+00:00` **progress** unit=`graph` payload=`{"phase":"run_start","user_identity":"高职院校教务处教师"}` id=`72172421` parent=`-`
  2. `2026-07-23T12:18:27.876780+00:00` **node_start** unit=`framework_orchestrator` payload=`{"step":1}` id=`ae0dca8f` parent=`72172421`
  3. `2026-07-23T12:18:27.881446+00:00` **llm_config_used** unit=`framework_orchestrator` payload=`{"unit":"framework_orchestrator","model":"fake-llm","base_url":"fake://"}` id=`e77973fa` parent=`ae0dca8f`
  4. `2026-07-23T12:18:27.881503+00:00` **state_snapshot** unit=`framework_orchestrator` payload=`{"status":"FRAMEWORK_BUILDING","iteration_round":0,"chapter_total":2,"chapters_completed":0,"material_count":0,"citation_retry_count":0,"citation_warning_count":0}` id=`70d640b3` parent=`ae0dca8f`
  5. `2026-07-23T12:18:27.881920+00:00` **progress** unit=`framework_orchestrator` payload=`{"chapters_completed":0,"chapter_total":2,"iteration_round":0,"status":"FRAMEWORK_BUILDING"}` id=`e784a33e` parent=`ae0dca8f`
  6. `2026-07-23T12:18:27.882115+00:00` **node_end** unit=`framework_orchestrator` payload=`{"step":1}` id=`deca0267` parent=`ae0dca8f`
  7. `2026-07-23T12:18:27.882190+00:00` **node_start** unit=`reference_orchestrator` payload=`{"chapter_id":"ch1","step":2}` id=`e75ab8a4` parent=`72172421`
  8. `2026-07-23T12:18:27.882244+00:00` **node_start** unit=`reference_orchestrator` payload=`{"chapter_id":"ch2","step":2}` id=`fda3bd02` parent=`72172421`
  9. `2026-07-23T12:18:27.884087+00:00` **subagent_start** unit=`search_agent` payload=`{"unit":"search_agent","chapter_id":"ch2","mode":null}` id=`27184cec` parent=`fda3bd02`
 10. `2026-07-23T12:18:27.884745+00:00` **subagent_end** unit=`search_agent` payload=`{"unit":"search_agent","chapter_id":"ch2","mode":null}` id=`4c4c59af` parent=`27184cec`
 11. `2026-07-23T12:18:27.885406+00:00` **subagent_start** unit=`search_agent` payload=`{"unit":"search_agent","chapter_id":"ch1","mode":null}` id=`1fa77f46` parent=`e75ab8a4`
 12. `2026-07-23T12:18:27.885603+00:00` **subagent_end** unit=`search_agent` payload=`{"unit":"search_agent","chapter_id":"ch1","mode":null}` id=`ff79d827` parent=`1fa77f46`
 13. `2026-07-23T12:18:27.888088+00:00` **llm_config_used** unit=`reference_orchestrator` payload=`{"unit":"reference_orchestrator"}` id=`9ff8c74a` parent=`fda3bd02`
 14. `2026-07-23T12:18:27.888139+00:00` **state_snapshot** unit=`reference_orchestrator` payload=`{"status":"REFERENCE_FETCHING","iteration_round":0,"chapter_total":2,"chapters_completed":0,"material_count":1,"citation_retry_count":0,"citation_warning_count":0}` id=`0e958d7c` parent=`fda3bd02`
 15. `2026-07-23T12:18:27.888176+00:00` **progress** unit=`reference_orchestrator` payload=`{"chapters_completed":0,"chapter_total":2,"iteration_round":0,"status":"REFERENCE_FETCHING"}` id=`b9ac1333` parent=`fda3bd02`
 16. `2026-07-23T12:18:27.888472+00:00` **node_end** unit=`reference_orchestrator` payload=`{"step":2}` id=`25079d4f` parent=`fda3bd02`
 17. `2026-07-23T12:18:27.888545+00:00` **llm_config_used** unit=`reference_orchestrator` payload=`{"unit":"reference_orchestrator"}` id=`ce56550b` parent=`fda3bd02`
 18. `2026-07-23T12:18:27.888638+00:00` **state_snapshot** unit=`reference_orchestrator` payload=`{"status":"REFERENCE_FETCHING","iteration_round":0,"chapter_total":2,"chapters_completed":0,"material_count":2,"citation_retry_count":0,"citation_warning_count":0}` id=`70824558` parent=`fda3bd02`
 19. `2026-07-23T12:18:27.888811+00:00` **progress** unit=`reference_orchestrator` payload=`{"chapters_completed":0,"chapter_total":2,"iteration_round":0,"status":"REFERENCE_FETCHING"}` id=`284f5159` parent=`fda3bd02`
 20. `2026-07-23T12:18:27.889317+00:00` **node_end** unit=`reference_orchestrator` payload=`{"step":2}` id=`a72cd7da` parent=`e75ab8a4`
 21. `2026-07-23T12:18:27.891288+00:00` **node_start** unit=`chapter_drafter` payload=`{"chapter_id":"ch1","step":4}` id=`8c4d363f` parent=`72172421`
 22. `2026-07-23T12:18:27.891587+00:00` **node_start** unit=`chapter_drafter` payload=`{"chapter_id":"ch2","step":4}` id=`5fd4e583` parent=`72172421`
 23. `2026-07-23T12:18:27.893849+00:00` **subagent_start** unit=`rewriter_loop` payload=`{"unit":"rewriter_loop","chapter_id":"ch2","mode":"draft"}` id=`51b0dd8e` parent=`5fd4e583`
 24. `2026-07-23T12:18:27.894395+00:00` **subagent_start** unit=`rewriter_loop` payload=`{"unit":"rewriter_loop","chapter_id":"ch1","mode":"draft"}` id=`a6306d93` parent=`8c4d363f`
 25. `2026-07-23T12:18:27.895474+00:00` **progress** unit=`rewriter_loop` payload=`{"unit":"rewriter_loop","chapter_id":"ch1","mode":"draft","step":"llm_call_start","call":"draft"}` id=`d3ed7867` parent=`a6306d93`
 26. `2026-07-23T12:18:27.895742+00:00` **progress** unit=`rewriter_loop` payload=`{"unit":"rewriter_loop","chapter_id":"ch2","mode":"draft","step":"llm_call_start","call":"draft"}` id=`48d4f281` parent=`51b0dd8e`
 27. `2026-07-23T12:18:27.903911+00:00` **progress** unit=`rewriter_loop` payload=`{"unit":"rewriter_loop","chapter_id":"ch1","mode":"draft","step":"llm_call_end","call":"draft","attempts":1,"text_chars":43,"degraded":false}` id=`ab574069` parent=`a6306d93`
 28. `2026-07-23T12:18:27.903999+00:00` **subagent_end** unit=`rewriter_loop` payload=`{"unit":"rewriter_loop","chapter_id":"ch1","mode":"draft"}` id=`fadb8d5a` parent=`a6306d93`
 29. `2026-07-23T12:18:27.904574+00:00` **subagent_start** unit=`chapter_reviewer` payload=`{"unit":"chapter_reviewer","chapter_id":"ch1","mode":"review"}` id=`224cfe57` parent=`5fd4e583`
 30. `2026-07-23T12:18:27.910864+00:00` **progress** unit=`rewriter_loop` payload=`{"unit":"rewriter_loop","chapter_id":"ch2","mode":"draft","step":"llm_call_end","call":"draft","attempts":1,"text_chars":60,"degraded":false}` id=`40cce4d4` parent=`51b0dd8e`
 31. `2026-07-23T12:18:27.910944+00:00` **subagent_end** unit=`rewriter_loop` payload=`{"unit":"rewriter_loop","chapter_id":"ch2","mode":"draft"}` id=`fd3c6264` parent=`51b0dd8e`
 32. `2026-07-23T12:18:27.911493+00:00` **subagent_start** unit=`chapter_reviewer` payload=`{"unit":"chapter_reviewer","chapter_id":"ch2","mode":"review"}` id=`9454ac78` parent=`5fd4e583`
 33. `2026-07-23T12:18:27.926632+00:00` **progress** unit=`chapter_reviewer` payload=`{"unit":"chapter_reviewer","chapter_id":"ch1","mode":"review","step":"lint_done","violations":0}` id=`58f6470a` parent=`224cfe57`
 34. `2026-07-23T12:18:27.926699+00:00` **progress** unit=`chapter_reviewer` payload=`{"unit":"chapter_reviewer","chapter_id":"ch1","mode":"review","step":"llm_call_start","call":"audit"}` id=`1363a60d` parent=`224cfe57`
 35. `2026-07-23T12:18:27.941882+00:00` **progress** unit=`chapter_reviewer` payload=`{"unit":"chapter_reviewer","chapter_id":"ch2","mode":"review","step":"lint_done","violations":0}` id=`f9437ef5` parent=`9454ac78`
 36. `2026-07-23T12:18:27.942682+00:00` **progress** unit=`chapter_reviewer` payload=`{"unit":"chapter_reviewer","chapter_id":"ch2","mode":"review","step":"llm_call_start","call":"audit"}` id=`163276d2` parent=`9454ac78`
 37. `2026-07-23T12:18:27.950989+00:00` **progress** unit=`chapter_reviewer` payload=`{"unit":"chapter_reviewer","chapter_id":"ch1","mode":"review","step":"llm_call_end","call":"audit","attempts":1,"degraded":false}` id=`b4498107` parent=`224cfe57`
 38. `2026-07-23T12:18:27.951064+00:00` **progress** unit=`chapter_reviewer` payload=`{"unit":"chapter_reviewer","chapter_id":"ch1","mode":"review","step":"audit_done","issues":0,"degraded":false}` id=`701b4220` parent=`224cfe57`
 39. `2026-07-23T12:18:27.951114+00:00` **progress** unit=`chapter_reviewer` payload=`{"unit":"chapter_reviewer","chapter_id":"ch1","mode":"review","step":"revision_note_done","violations":0,"passed":true}` id=`7b550c78` parent=`224cfe57`
 40. `2026-07-23T12:18:27.951527+00:00` **subagent_end** unit=`chapter_reviewer` payload=`{"unit":"chapter_reviewer","chapter_id":"ch1","mode":"review"}` id=`c191216c` parent=`224cfe57`
 41. `2026-07-23T12:18:27.960826+00:00` **progress** unit=`chapter_reviewer` payload=`{"unit":"chapter_reviewer","chapter_id":"ch2","mode":"review","step":"llm_call_end","call":"audit","attempts":1,"degraded":false}` id=`dd5d9020` parent=`9454ac78`
 42. `2026-07-23T12:18:27.961015+00:00` **progress** unit=`chapter_reviewer` payload=`{"unit":"chapter_reviewer","chapter_id":"ch2","mode":"review","step":"audit_done","issues":0,"degraded":false}` id=`73058422` parent=`9454ac78`
 43. `2026-07-23T12:18:27.961071+00:00` **progress** unit=`chapter_reviewer` payload=`{"unit":"chapter_reviewer","chapter_id":"ch2","mode":"review","step":"revision_note_done","violations":0,"passed":true}` id=`66a9f65d` parent=`9454ac78`
 44. `2026-07-23T12:18:27.961116+00:00` **subagent_end** unit=`chapter_reviewer` payload=`{"unit":"chapter_reviewer","chapter_id":"ch2","mode":"review"}` id=`72d54c1b` parent=`9454ac78`
 45. `2026-07-23T12:18:27.964263+00:00` **llm_config_used** unit=`chapter_drafter` payload=`{"unit":"chapter_drafter"}` id=`ead4986c` parent=`5fd4e583`
 46. `2026-07-23T12:18:27.964418+00:00` **state_snapshot** unit=`chapter_drafter` payload=`{"status":"ARTICLE_WRITING","iteration_round":0,"chapter_total":2,"chapters_completed":1,"material_count":2,"citation_retry_count":0,"citation_warning_count":0}` id=`bd79319c` parent=`5fd4e583`
 47. `2026-07-23T12:18:27.964474+00:00` **progress** unit=`chapter_drafter` payload=`{"chapters_completed":1,"chapter_total":2,"iteration_round":0,"status":"ARTICLE_WRITING"}` id=`111a033f` parent=`5fd4e583`
 48. `2026-07-23T12:18:27.965419+00:00` **node_end** unit=`chapter_drafter` payload=`{"step":4}` id=`9fd9e362` parent=`8c4d363f`
 49. `2026-07-23T12:18:27.967355+00:00` **llm_config_used** unit=`chapter_drafter` payload=`{"unit":"chapter_drafter"}` id=`58da15f3` parent=`5fd4e583`
 50. `2026-07-23T12:18:27.967583+00:00` **state_snapshot** unit=`chapter_drafter` payload=`{"status":"ARTICLE_WRITING","iteration_round":0,"chapter_total":2,"chapters_completed":2,"material_count":2,"citation_retry_count":0,"citation_warning_count":0}` id=`e3576595` parent=`5fd4e583`
 51. `2026-07-23T12:18:27.968046+00:00` **progress** unit=`chapter_drafter` payload=`{"chapters_completed":2,"chapter_total":2,"iteration_round":0,"status":"ARTICLE_WRITING"}` id=`0d0367f2` parent=`5fd4e583`
 52. `2026-07-23T12:18:27.968285+00:00` **node_end** unit=`chapter_drafter` payload=`{"step":4}` id=`08079822` parent=`5fd4e583`
 53. `2026-07-23T12:18:27.969446+00:00` **node_start** unit=`document_reviewer` payload=`{"step":5}` id=`ca37ad61` parent=`72172421`
 54. `2026-07-23T12:18:27.975557+00:00` **llm_config_used** unit=`document_reviewer` payload=`{"unit":"document_reviewer","model":"fake-llm","base_url":"fake://"}` id=`eabff8a2` parent=`ca37ad61`
 55. `2026-07-23T12:18:27.975811+00:00` **state_snapshot** unit=`document_reviewer` payload=`{"status":"AWAIT_USER_REVIEW","iteration_round":0,"chapter_total":2,"chapters_completed":2,"material_count":2,"citation_retry_count":0,"citation_warning_count":0}` id=`26ba9470` parent=`ca37ad61`
 56. `2026-07-23T12:18:27.977458+00:00` **progress** unit=`document_reviewer` payload=`{"chapters_completed":2,"chapter_total":2,"iteration_round":0,"status":"AWAIT_USER_REVIEW"}` id=`52cc1cb1` parent=`ca37ad61`
 57. `2026-07-23T12:18:27.977729+00:00` **branch_taken** unit=`document_reviewer` payload=`{"from":"document_reviewer","to":"human_review_gate","reason":"篇级终审通过或重试超限，进入人工中断点"}` id=`3bfce7fe` parent=`ca37ad61`
 58. `2026-07-23T12:18:27.977856+00:00` **node_end** unit=`document_reviewer` payload=`{"step":5}` id=`3ea4f1d0` parent=`ca37ad61`
 59. `2026-07-23T12:18:27.977913+00:00` **node_start** unit=`human_review_gate` payload=`{"step":6}` id=`254dda0d` parent=`72172421`
 60. `2026-07-23T12:18:27.981187+00:00` **gate_blocked** unit=`human_review_gate` payload=`{"iteration_round":0,"chapter_ids":["ch1","ch2"],"citation_warnings":[],"review_warnings":[]}` id=`323834b5` parent=`254dda0d`
 61. `2026-07-23T12:18:27.981268+00:00` **node_end** unit=`human_review_gate` payload=`{"step":6,"interrupted":true}` id=`361ee64f` parent=`254dda0d`

### 业务 SSE 通道

  1. `2026-07-23 20:18:27` **status** data=`{"status":"FRAMEWORK_BUILDING","iteration_round":0,"node":"framework_orchestrator"}`
  2. `2026-07-23 20:18:27` **status** data=`{"status":"REFERENCE_FETCHING","iteration_round":0,"node":"reference_orchestrator"}`
  3. `2026-07-23 20:18:27` **status** data=`{"status":"REFERENCE_FETCHING","iteration_round":0,"node":"reference_orchestrator"}`
  4. `2026-07-23 20:18:27` **status** data=`{"status":"ARTICLE_WRITING","iteration_round":0,"node":"chapter_drafter"}`
  5. `2026-07-23 20:18:27` **status** data=`{"status":"ARTICLE_WRITING","iteration_round":0,"node":"chapter_drafter"}`
  6. `2026-07-23 20:18:27` **status** data=`{"status":"AWAIT_USER_REVIEW","iteration_round":0,"node":"document_reviewer"}`
  7. `2026-07-23 20:18:28` **review_required** data=`{"iteration_round":0,"chapter_ids":["ch1","ch2"],"citation_warnings":[],"review_warnings":[]}`

## 每章中间产物

说明：事件信封按设计绝不携带正文全文，且服务未暴露逐章 state 的
REST 读接口，故本节由「事件流推导 + 人工中断点时的整篇渲染快照 +
定稿载荷」三路拼合——草稿正文取自首次人工中断点的书目接口渲染
（已统一重编号），self_check 的 citations_ok 与 issues 按各写作调用的
lint_done / audit_done / revise_triggered 进度事件推导（明细文本未经
REST 暴露时以计数呈现），是否触发修订以 revise_triggered 事件为准；
触发修订时末次 lint_done / audit_done 为修后复检计数（ADR-0004），
citations_ok 按末次（终态）计数推导。

（运行未到定稿，无法枚举章节。）

## 逐章 state 演进快照

快照来源：graph_event 通道的 state_snapshot 事件（每个超步节点更新后
各发布一条，载荷为纯计数元数据）。「已完成章节」由 rewriter_loop 的
subagent_end 事件顺序累积推导。

| # | 时间 | 单元 | 已完成章节 | 草稿数/总章数 | 引文库条数 | 迭代轮次 | 状态 |
|---|------|------|------------|---------------|------------|----------|------|
| 1 | 2026-07-23T12:18:27.881503+00:00 | framework_orchestrator | — | 0/2 | 0 | 0 | FRAMEWORK_BUILDING |
| 2 | 2026-07-23T12:18:27.888139+00:00 | reference_orchestrator | — | 0/2 | 1 | 0 | REFERENCE_FETCHING |
| 3 | 2026-07-23T12:18:27.888638+00:00 | reference_orchestrator | — | 0/2 | 2 | 0 | REFERENCE_FETCHING |
| 4 | 2026-07-23T12:18:27.964418+00:00 | chapter_drafter | ch1、ch2 | 1/2 | 2 | 0 | ARTICLE_WRITING |
| 5 | 2026-07-23T12:18:27.967583+00:00 | chapter_drafter | ch1、ch2 | 2/2 | 2 | 0 | ARTICLE_WRITING |
| 6 | 2026-07-23T12:18:27.975811+00:00 | document_reviewer | ch1、ch2 | 2/2 | 2 | 0 | AWAIT_USER_REVIEW |

## 修订与终审

- 第 1 次人工中断点（`2026-07-23 20:18:28`）：
  - 中断载荷：`{"iteration_round":0,"chapter_ids":["ch1","ch2"],"citation_warnings":[],"review_warnings":[]}`
- 终审结果：未到定稿

## 最终产物

### 整篇文章（统一重编号后）

（运行未到定稿。）
### 统一重编号书目（gbt7714）

（未捕获该格式的渲染结果。）
### 书目（markdown）

（未捕获该格式的渲染结果。）
