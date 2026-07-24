# HypoArgus-Writer 对外接口文档

面向外部调用方（如 Java 服务端）的 REST + SSE 接口契约。
接口实现见 `src/service/app.py`；事件信封契约见 `src/service/event_envelope.py`。

## 1. 总览

- 协议：普通 HTTP REST + Server-Sent Events（SSE），**不是** LangGraph SDK/Server 协议，任意 HTTP 客户端可直接对接。
- 编码：请求与响应均为 UTF-8 JSON；SSE 为 `text/event-stream`。
- 鉴权：**无**。`session_id` 由调用方生成并透传，本系统只透传不校验，用于调用方侧的会话隔离与事件过滤。
- 缺省服务地址：`http://<host>:8000`。

### 1.1 核心概念

| 概念 | 说明 |
|---|---|
| `thread_id` | 任务唯一标识（32 位 hex UUID），创建任务时返回，后续所有接口以它寻址 |
| `execution_trace_id` | 一次任务的追踪标识，出现在可视化事件信封的 `trace_id` 字段 |
| 任务状态 | `IDLE` / `FRAMEWORK_BUILDING` / `REFERENCE_FETCHING` / `ARTICLE_WRITING` / `CITATION_CHECKING` / `AWAIT_USER_REVIEW` / `FINISHED` / `ERROR_FAILED` |
| 人工审阅 | 任务写完一轮后停在人工中断点（`awaiting_review=true`），调用方提交 `finalize`（定稿）或 `revise`（携修订意见再迭代一轮），可无限迭代；`revise` 触及超过大纲一半章节时系统先回显解析清单，需提交 `confirm` 确认后才执行 |
| 检查点 | 每个关键步骤持久化到 Postgres；支持崩溃恢复与回滚到任意历史检查点 |
| mock 任务 | `POST /tasks` 带 `mock:true` 触发的确定性 mock 栈任务，`thread_id` 带 `mock-` 前缀，与真任务共享同一检查点存档器；状态/产物/审阅包/书目响应均带 `mock:true` 标记，供集成测试与联调秒回审阅门 |

### 1.2 典型调用时序

```
Java 端                                  HypoArgus-Writer
  │ POST /tasks                            │
  │──────────────────────────────────────▶│ 201 {thread_id, execution_trace_id}
  │ GET /tasks/{id}/stream  (SSE 长连接)   │
  │──────────────────────────────────────▶│
  │        ◀── event: status  (多次，各阶段推进)
  │        ◀── event: review_pack_ready （审阅包摘要 + pack_version，丢了靠 REST 取）
  │        ◀── event: review_required     （任务停在人工中断点，纯路由元数据）
  │ GET /tasks/{id}/review              （取审阅包全文：大纲/正文/警告/台账/引文库）
  │──────────────────────────────────────▶│ 200 {pack_version, outline, chapters, ...}
  │ POST /tasks/{id}/review {action:"revise", feedback:"..."}
  │──────────────────────────────────────▶│ 202
  │        ◀── event: status ...          （新一轮迭代）
  │        ◀── event: review_required
  │ POST /tasks/{id}/review {action:"finalize"}
  │──────────────────────────────────────▶│ 202
  │        ◀── event: finalized           （含全文；随后 SSE 正常结束）
  │ GET /tasks/{id}/bibliography?format=gbt7714
  │──────────────────────────────────────▶│ 200 重编号正文 + 书目
```

### 1.3 错误响应

所有错误响应体统一为：

```json
{ "detail": "错误说明文本" }
```

| 状态码 | 含义 |
|---|---|
| 400 | 参数非法（书目格式未注册、事件类型过滤值非法） |
| 404 | 任务 / 检查点不存在 |
| 409 | 当前状态不允许该操作（任务运行中、未停在中断点、尚无正文） |
| 422 | 请求体不符合契约（含 FastAPI 标准校验错误、`revise` 缺 feedback、检索任务违反引擎入参契约） |
| 503 | 检索通道 / LLM 配置缺失，独立检索暂不可用 |

## 2. REST 接口

### 2.1 创建任务

`POST /tasks` → `201 Created`

创建写作任务并立即异步启动首跑。

请求体：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `user_intent` | string | 是 | 写作意图，不允许空白 |
| `user_identity` | string | 否 | 用户身份标识，缺省 `""` |
| `session_id` | string | 否 | 调用方会话标识，缺省 `""`，只透传 |
| `mock` | boolean | 否 | 缺省 `false`；`true` 时走确定性 mock 栈，秒回审阅门，`thread_id` 带 `mock-` 前缀，供集成测试与联调 |

```json
{ "user_intent": "写一篇论证国产数据库替代可行性的行业白皮书", "user_identity": "u-1001", "session_id": "sess-abc" }
```

响应体：

```json
{ "thread_id": "9f3c...32位hex", "execution_trace_id": "1d2e...32位hex" }
```

`mock:true` 时响应体额外带 `"mock": true`，且 `thread_id` 形如 `mock-<32位hex>`。

### 2.2 查询任务状态

`GET /tasks/{thread_id}` → `200 OK`

```json
{
  "thread_id": "9f3c...",
  "status": "AWAIT_USER_REVIEW",
  "iteration_round": 1,
  "awaiting_review": true,
  "running": false,
  "mock": false
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `status` | string | 见 1.1 状态枚举 |
| `iteration_round` | int | 当前迭代轮次，从 0 起 |
| `awaiting_review` | bool | 是否停在人工中断点（true 时才能提交审阅） |
| `running` | bool | 是否有图运行正在进行 |
| `mock` | bool | 是否 mock 任务（`thread_id` 带 `mock-` 前缀），真任务为 `false` |

错误：任务不存在 → 404。

### 2.3 提交人工审阅

`POST /tasks/{thread_id}/review` → `202 Accepted`

从人工中断点恢复运行。
仅当 `awaiting_review=true` 且 `running=false` 时可调，否则 409。

请求体：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `action` | string | 是 | `"finalize"`（定稿）、`"revise"`（修订）或 `"confirm"`（确认大扇出修订清单） |
| `feedback` | string | revise 时必填 | 自然语言修订意见，不允许空白 |

```json
{ "action": "revise", "feedback": "引言口吻克制些；第二章补充行业数据佐证" }
```

响应体：

```json
{ "thread_id": "9f3c...", "action": "revise" }
```

语义：`revise` 会由 LLM 把意见解析为逐章修订指令并再迭代一轮，之后重新停在中断点；`finalize` 直接进入 `FINISHED` 并在业务 SSE 推送 `finalized` 事件。
若意见解析不出有效指令，系统不报错，而是携 `error` 字段重新推送 `review_required` 事件，调用方重新提交即可。

停门后审阅所需全文（大纲/各章正文/引文警告/篇级 warn/修订台账/引文库）一次取齐见 §2.9 `GET /tasks/{id}/review`；`review_required` 事件只携路由元数据（`chapter_ids`/`pending_confirmation`/`clarification_questions`/`iteration_round`/`error`），不带正文与警告全文。

修订指令定位增强下的三种重新中断（均通过重新推送 `review_required` 事件呈现，调用方按载荷字段区分）：

- 意见引用了正文原文时，系统先做确定性子串匹配直达目标章；未命中回退 LLM 判断到章，对调用方透明。
- 解析出的指令触及超过大纲一半的章节（含全局意见扇出为逐章指令的情形）时，事件携 `pending_confirmation` 解析清单重新中断；调用方提交 `{"action": "confirm"}` 确认执行，或改提 `revise` / `finalize`。
- 意见含混或引文定位失败时，事件携 `clarification_questions` 回问问题重新中断，系统不猜测；调用方补充说明后重新提交 `revise`。

`confirm` 仅在最近一次 `review_required` 事件携 `pending_confirmation` 时有意义，否则系统携 `error` 重新推送 `review_required`。

### 2.4 崩溃恢复

`POST /tasks/{thread_id}/resume` → `200 OK`

服务重启或运行中断后，按 Postgres 检查点恢复任务。

请求体：

```json
{ "session_id": "sess-abc" }
```

`session_id` 可选；传非空值会覆盖登记值。

响应体：

```json
{ "thread_id": "9f3c...", "status": "AWAIT_USER_REVIEW" }
```

三种情形：

1. 停在中断点：不重跑图，只在两条 SSE 通道补发 `gate_blocked` 与 `review_required` 事件。
2. 中途被杀：从最近检查点继续驱动。
3. 已到终态：仅重建登记，业务 SSE 通道随即正常收尾。

错误：无检查点 → 404；已有运行进行中 → 409。

### 2.5 回滚到历史检查点

`POST /tasks/{thread_id}/rollback` → `202 Accepted`

从指定历史检查点分叉重放；重放到人工审阅门会重新中断，调用方从该历史版本继续迭代。

请求体：

```json
{ "checkpoint_id": "1ef3..." }
```

响应体：

```json
{ "thread_id": "9f3c...", "checkpoint_id": "1ef3..." }
```

错误：检查点不存在 → 404；运行中 → 409。

### 2.6 检查点清单

`GET /tasks/{thread_id}/checkpoints` → `200 OK`

返回新到旧的检查点元数据数组（绝不含正文），供回滚选点。

```json
[
  {
    "checkpoint_id": "1ef3...",
    "ts": "2026-07-19T08:00:00.000000+00:00",
    "status": "AWAIT_USER_REVIEW",
    "iteration_round": 1,
    "next": ["human_review_gate"]
  }
]
```

### 2.7 渲染最终交付（重编号正文 + 书目）

`GET /tasks/{thread_id}/bibliography?format=gbt7714` → `200 OK`

`format` 可选值：`gbt7714`（缺省）、`apa`、`markdown`。

```json
{
  "thread_id": "9f3c...",
  "format": "gbt7714",
  "chapters": [
    { "chapter_id": "ch1", "text": "正文……角标已重编号为 [1] 形式" }
  ],
  "bibliography": [
    {
      "index": 1,
      "material_id": "m_0123456789ABCDEFGHJKMNPQRS",
      "material": {
        "id": "m_0123456789ABCDEFGHJKMNPQRS",
        "hypothesis_id": "h1",
        "chapter_id": "ch1",
        "source": "来源名称",
        "url": "https://example.com/evidence",
        "source_kind": "web",
        "source_ref": {
          "url": "https://example.com/evidence",
          "content_fingerprint": "..."
        },
        "excerpt": "证据摘录……",
        "relevance_score": 0.9,
        "verdict": "pass"
      },
      "text": "[1] 来源[EB/OL]. https://example.com/evidence."
    }
  ]
}
```

错误：格式未注册 → 400；尚无章节正文（框架 / 检索阶段）→ 409。

说明：`finalized` SSE 事件中的正文保留原始素材 id 角标；本接口返回按出现顺序重编号后的正文与配套书目，是推荐的最终产物获取方式。
`bibliography[].index` 是服务端生成的最终展示编号，调用方不得自行重编号。
`bibliography[].material_id` 为兼容旧消费者保留，恒等于 `bibliography[].material.id`。
`bibliography[].material` 使用统一 `Material` 契约，包含稳定不透明 `id`、可选 `url`、三值 `source_kind` 与可选 `source_ref`。
`source_ref` 是真实来源定位；`Material.id` 只承担正文角标和跨接口关联身份，不承载来源定位明文。
mock 任务与真实任务的 `bibliography[].material` 形状相同。

### 2.8 独立检索

`POST /retrieval` → `200 OK`

阻塞式独立检索：提交一章假说列表，同步等待检索完成并返回素材与诊断，不启动写作任务。
与写作主流程使用同一套任务包/结果契约与同一 search_agent 实例，两处调用行为一致。

请求体：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `chapter_id` | string | 是 | 章节标识，不允许空白，进事件上下文；素材 id 由稳定来源定位确定性派生，不再嵌入章节标识 |
| `points` | array | 否 | 论点列表，缺省 `[]`；条目为 `{id, text}`（`id` 与 `text` 不允许空白），与假说一并聚合进查询构造 |
| `hypotheses` | array | 是 | 假说列表，不允许为空；条目为 `{id, text, refute_condition}`，`id` 与 `text` 不允许空白，`refute_condition` 非空白时驱动反向检索 |
| `genre` | string | 否 | 品类（检索范围提示），缺省 `""` |
| `existing_materials_digest` | string | 否 | 既有素材摘要（供引擎规避重复素材），缺省 `""` |
| `session_id` | string | 否 | 调用方会话标识，缺省 `""`，进度事件按其入信封供 `/graph_events` 过滤订阅 |

```json
{
  "chapter_id": "ch1",
  "points": [
    { "id": "p1", "text": "国产数据库替代能力的现状与边界" }
  ],
  "hypotheses": [
    { "id": "h1", "text": "国产数据库在核心交易场景已具备替代能力", "refute_condition": "近两年存在因性能问题回迁的公开案例" }
  ],
  "session_id": "sess-abc"
}
```

响应体：

```json
{
  "materials": [
    {
      "id": "m_0123456789ABCDEFGHJKMNPQRS",
      "hypothesis_id": "h1",
      "source": "来源名称",
      "url": "https://example.com/evidence",
      "source_kind": "web",
      "source_ref": {
        "url": "https://example.com/evidence",
        "content_fingerprint": "..."
      },
      "excerpt": "证据摘录……",
      "relevance_score": 0.9,
      "verdict": "pass"
    }
  ],
  "diagnostics": { "total_elapsed_ms": 1234, "call_counts": { "web_search": 2 }, "weak_evidence_count": 1, "pass_below_threshold": { "pass_count": 2, "threshold": 3 } }
}
```

| 字段 | 说明 |
|---|---|
| `materials[].id` | 正文可见素材 id，形态固定为 `m_<26位CrockfordBase32>`；该值不包含章节 id、假说 id、上游 citation id 或来源定位明文 |
| `materials[].hypothesis_id` | 素材回链的假说 id |
| `materials[].url` | 来源链接；仅联网来源必带，知识库与结构化来源可为 `null` |
| `materials[].source_kind` | 来源通道三值：`web` / `knowledge_base` / `structured_data` |
| `materials[].source_ref` | 真实来源定位；web 通常含 `url`，知识库通常含 `knowledge_id`/`file_id`/`chunk_id`，结构化数据通常含 `scenario_key`/`dataset_id`/`query_execution_id` |
| `materials[].verdict` | 佐证强度三值：`pass`（强支撑，可作量化断言依据）/ `inconclusive`（弱佐证，近似命中/补充，仅作背景提示）/ `fail`（反例或不可用，供筛选审计）。**消费方须按三值处理** |
| `diagnostics` | 本次检索的诊断摘要（计数与耗时），与 `subagent_end` 事件携带的诊断同源；可含 `weak_evidence_count`（本章弱佐证条数）与 `pass_below_threshold`（pass 落库低于下限的薄弱章警告）|

进度事件：调用期间以请求中的 `session_id` 在 `/graph_events` 通道发布 `subagent_start` → `progress`（多条）→ `subagent_end` 事件链，`subagent_start` 为本次调用的根事件（`thread_id` 为空串）。

错误：请求体校验失败 → 422；检索任务违反引擎入参契约 → 422；检索通道 / LLM 配置缺失 → 503。

### 2.9 人工审阅包

`GET /tasks/{thread_id}/review` → `200 OK`

停在人工中断点时一次返回完整审阅包全文供调用方开展审阅；
未停在中断点返回 409（不返回半成品），任务不存在返回 404。
重复调用幂等（同检查点同 `pack_version`）。

响应体（六类内容齐备 + 轮次指纹）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `pack_version` | string | 轮次指纹：六类内容 + 迭代轮次的 sha256 前 16 位；同状态恒同，修订再停门后内容变化即指纹变化 |
| `iteration_round` | int | 当前迭代轮次，从 0 起 |
| `outline` | array | 当前轮大纲（`ChapterSpec`：`id`/`title`/`subsections`/`chapter_type`/`planned_summary`/`points`[含 `hypotheses`]） |
| `chapters` | array | 各章正文：`{chapter_id, text, summary}`，`text` 含原位素材 id 角标 |
| `citation_warnings` | array[string] | 引文重试超限的未决警告，交人工裁决 |
| `review_warnings` | array[string] | 篇级评审 warn 级提示（章间衔接/口径统一/跨章重复），不打回、不影响重试 |
| `revision_ledger` | array | 修订台账（`RevisionRound`：`round_no`/`raw_feedback`/`directives`/`digest`），全量持久化 |
| `citation_library` | array | 引文库素材全文（`Material`：`id`/`hypothesis_id`/`chapter_id`/`source`/`url`/`source_kind`/`source_ref`/`excerpt`/`relevance_score`/`verdict`） |

调用方在收到 SSE `review_pack_ready` 摘要或 `review_required` 路由信号后取此全文审阅；
SSE 摘要丢了靠此 REST 对账重取。
`review_required` 只携路由元数据，引文警告与篇级 warn 全文只走本接口。

### 2.10 运行中产物快照

`GET /tasks/{thread_id}/products` → `200 OK`

任意状态可调的只读检查点快照：目录/假说/各章素材/已完成章正文，章级粒度。
纯只读检查点 state，不引入新状态、不加写路径；SSE 丢帧（`product` / `content_delta`）靠此 REST 对账重取。
与 §2.2 状态查询同属只读探态，但本接口额外给运行中内容。
与 §2.9 审阅包携带的是同一份检查点 state，但分形不同：§2.9 把大纲（`outline`，含 `points`/`hypotheses`）、各章正文（`chapters`，仅 `chapter_id`/`text`/`summary`）、引文库（`citation_library`，扁平素材）分字段平铺且仅在停门可取；本接口把它们按章聚合成 `chapters[]`（每章内嵌 `points`/`materials`/`draft`）。两者取数边界对比：

- §2.9 `/review` 仅 `awaiting_review=true` 可取，否则 409（不返回半成品）；
- 本接口**任意状态可取**，未完成部分由字段值表达边界——章未检索时 `materials` 为空、未写完时 `draft` 为 `null`、刚创建尚无大纲时 `chapters` 为空列表，**不返回 409**。

任务不存在 → 404。
mock 任务响应带 `mock: true`，真任务 `mock: false`（见 §2.11）。

响应体：

```json
{
  "thread_id": "9f3c...",
  "status": "ARTICLE_WRITING",
  "iteration_round": 0,
  "mock": false,
  "chapters": [
    {
      "chapter_id": "ch1",
      "title": "国产数据库替代的现状",
      "subsections": ["1.1 渗透率", "1.2 替代路径"],
      "chapter_type": "骨架章标题原文或 null",
      "planned_summary": "本章预判一句话概要",
      "points": [
        {
          "id": "p1",
          "text": "国产数据库替代能力的现状与边界",
          "hypotheses": [
            { "id": "h1", "text": "...", "refute_condition": "...", "angle": "假设" }
          ]
        }
      ],
      "materials": [
        {
          "id": "m_0123456789ABCDEFGHJKMNPQRS",
          "hypothesis_id": "h1",
          "chapter_id": "ch1",
          "source": "来源名称",
          "url": "https://example.com/evidence",
          "source_kind": "web",
          "source_ref": {
            "url": "https://example.com/evidence",
            "content_fingerprint": "..."
          },
          "excerpt": "证据摘录……",
          "relevance_score": 0.9,
          "verdict": "pass"
        }
      ],
      "draft": {
        "chapter_id": "ch1",
        "text": "正文……含原位素材 id 角标",
        "summary": "本章一句话摘要",
        "self_check": { "citations_ok": true, "issues": [] }
      }
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `status` / `iteration_round` / `mock` | string / int / bool | 与 §2.2 `GET /tasks/{id}` 同源（同检查点 state 派生） |
| `chapters` | array | 章级快照，逐章对齐大纲；刚创建尚无大纲时为空列表 |
| `chapters[].chapter_id` | string | 由 `ChapterSpec.id` 映射 |
| `chapters[].chapter_type` | string\|null | 骨架章标题原文；自由结构模式为 `null` |
| `chapters[].points` | array | `ArgumentPoint`：`{id, text, hypotheses:[...]}` |
| `chapters[].points[].hypotheses` | array | `Hypothesis`：`{id, text, refute_condition, angle}` |
| `chapters[].materials` | array | 该章已落库素材（按章分组）；未检索为空 |
| `chapters[].materials[]` | object | `Material` 全字段：`id`/`hypothesis_id`/`chapter_id`/`source`/`url`/`source_kind`/`source_ref`/`excerpt`/`relevance_score`/`verdict`；`id` 形态为 `m_<26位CrockfordBase32>`；`url` 仅联网来源必带，余者可为 `null`；真实定位见 `source_ref`；`source_kind` 三值 `web`/`knowledge_base`/`structured_data`；`verdict` 三值 `pass`/`inconclusive`/`fail` |
| `chapters[].draft` | object\|null | `ChapterDraft`：`{chapter_id, text, summary, self_check}`；未写完为 `null`，调用方据此判本章是否已落正文 |
| `chapters[].draft.self_check` | object | `SelfCheck`：`{citations_ok: bool, issues: [string]}` |

说明：本接口 `chapters[].draft` 与 §3.1 `chapter_ready` 事件的 `draft` 载荷严格同构（`{chapter_id, text, summary, self_check}`），`chapters[].materials` 与 `materials_ready` 事件的 `materials` 同构（`Material` 全字段），章级骨架（`title`/`subsections`/`chapter_type`/`planned_summary`/`points`/`hypotheses`）与 `outline_ready` 事件的 `outline[]` 同构（仅章标识在事件载荷记为 `id`、本接口记为 `chapter_id`，同源 `ChapterSpec.id`）。
SSE 丢帧后按 REST 取回同等内容。
`draft.text` 含原位素材 id 角标（未重编号）；重编号正文 + 配套书目走 §2.7 `GET /tasks/{id}/bibliography`。

### 2.11 mock 档与清理策略

mock 档是确定性的「形如真」场景栈，供集成测试与联调秒回审阅门，无需真实模型链路。
mock 栈在 `create_app` 的 lifespan 期装配：FakeLLM（场景库提供顺序/键控应答）
+ 真改写器（`make_rewriter_loop`，逐字流与退化重试与真栈同口径）
+ 打桩 `search_agent`/`chapter_reviewer`（零 LLM、恒通过）；
与真栈共用同一 checkpointer，故 mock 任务的检查点、崩溃恢复、回滚与真任务同形。

场景库（`src/service/mock_scenarios.py`）单点登记四类分支覆盖：
`DEFAULT_SCENARIO`（多章大纲解析 / 角标正文 / 篇级 transition warn）、
`DEGRADATION_SCENARIO`（ch1 写作键控序列含 malformed JSON，配小 flush 阈值触发 attempt 1 失败、attempt 2 成功的退化重试）。
新增场景只在此单点登记，装配档只认 `MockScenario` 一个类型。

mock 任务的 `thread_id` 带 `mock-` 前缀，`TaskManager` 按前缀路由到 mock 图；
`GET /tasks/{id}`、`GET /tasks/{id}/products`、`GET /tasks/{id}/review`、
`GET /tasks/{id}/bibliography` 的响应均带 `mock: true` 标记，真任务带 `mock: false`。
mock 任务可在任意环境触发（含线上），已知并接受；其检查点与真任务混存于同一存档器。

生产环境定期清理 mock 检查点线程，避免 mock 残留占用存档空间：
脚本 `scripts/cleanup_mock_tasks.py` 读 `HYPOARGUS_PG_DSN` 连接 Postgres，
按 `thread_id LIKE 'mock-%'` 删除 `checkpoints` / `checkpoint_blobs` / `checkpoint_writes` 三表；
保留窗口由环境变量 `MOCK_CLEANUP_DAYS` 控制，缺省 7 天。
运行期清理亦可通过 `TaskManager.purge_mock_threads()` 触发（枚举存档器中 `mock-` 前缀 thread_id，逐个 `delete_thread` 并摘除内存登记）。

## 3. SSE 通道

两条通道，帧格式均为标准 SSE 三行：

```
id: <event_id>
event: <事件类型>
data: <JSON>

```

传输层基于 `sse-starlette` `EventSourceResponse`：长连接周期性收到 keepalive ping（缺省 15s，不被中间网关掐断）、客户端断开被服务端检测并停止推流、服务关停优雅关流。

事件 id 形如 `{epoch}-{seq}`：`epoch` 为进程（应用实例）启动标识，`seq` 在单流内单调递增。
`id` 行即 `Last-Event-ID` 续传凭据。

订阅语义（**取代旧的"订阅先全量回放"**）：

- 不带 `Last-Event-ID` 的新订阅**只收实时事件，不回放历史**。
- 带 `Last-Event-ID` 重连，只续推该 id 之后仍保留在服务端缓冲内的事件，不重复、不全文回放。
- 续传注定失效时（服务端世代切换即 `epoch` 不匹配，或所求位置已被淘汰），立即下发 `reconcile_required` 控制事件（载荷指明该走哪些 REST 口子对账），随后转实时推送，**绝不静默错位续推**。
- 进程重启后不回放：客户端收 `reconcile_required` 后靠 REST（status / bibliography / 检查点）对账。

建议客户端不设读超时（长连接），断线带 `Last-Event-ID` 重连续传。

**慢消费者背压（两级丢弃——信号必达，产物可丢可取）**：每订阅者一条有界队列（容量由环境变量 `SSE_MAX_QUEUE` 调，缺省 256）。慢消费者灌满时按两级丢弃挤位：

- 可丢级（`content_delta` 逐字帧、`product` 整块产物）队列满时丢最旧一条并累计 `dropped`；丢了靠 REST（`GET /tasks/{id}/products`、`/review`、`/bibliography`）对账重取。
- 不可丢控制信号（`review_required` / `finalized` / `error` / `reconcile_required`）满时挤掉可丢级帧为其让位；全队列无可丢级可挤时强制超容入队——信号体积极小、罕见，保信号必达，无内存风险。

正常速率消费者不丢不重。`dropped`（历史缓冲淘汰 + 队列丢弃累计）与 `subscriber_count` 经下方 stats 端点观测。

### 3.1 业务通道 `GET /tasks/{thread_id}/stream`

每任务一条流；不带 `Last-Event-ID` 的新订阅只收实时事件，带 `Last-Event-ID`
重连只续推该 id 之后的事件；任务定稿或失败后流**正常结束**。

`data` 统一结构：

```json
{ "event_id": "{epoch}-{seq}", "type": "...", "thread_id": "...", "ts": "UTC ISO8601", "data": { } }
```

| `type` | `data` 结构 | 说明 |
|---|---|---|
| `status` | `{"status": "...", "iteration_round": n, "node": "节点名"}` | 状态机推进 |
| `product` | `{"kind": "outline_ready"\|"materials_ready"\|"chapter_ready", ...}` | 结构化产物整块事件，按产出顺序推送；属可丢级，丢了靠 `GET /tasks/{id}/products` 对账重取（见下） |
| `content_delta` | `{"chapter_id": "...", "mode": "draft"\|"revise", "kind": "content"\|"thinking", "delta": "...", "attempt": <int>, "sequence": <int>}` | 写作中正文逐字增量（`kind=content` 为纯正文，非 JSON 语法碎片；思考开启时 `kind=thinking` 随正文一并逐字推送）；仅 writer draft/revise 产生，其他运行单元不逐字流。属可丢级，丢了不影响终态——`chapter_ready` 整块是持久锚，逐字流非持久化 |
| `review_required` | `{"iteration_round": n, "chapter_ids": [...], "error"?: "...", "clarification_questions"?: [...], "pending_confirmation"?: {...}}` | 停在人工中断点——纯到达信号，仅携路由元数据供调用方判分支：`error`（上次提交契约不符/解析失败，须重新提交）、`clarification_questions`（意见含混/引文定位失败的回问问题）、`pending_confirmation`（大扇出待确认清单：`{"affected_chapter_ids": [...], "total_chapters": n, "directives": [{"target_chapter_id", "type", "instruction"}]}`）按场景出现。引文警告/篇级 warn/章正文/素材全文不在本事件，走 `GET /tasks/{id}/review` |
| `reconcile_required` | `{"reason": "epoch_mismatch"\|"position_dropped"\|"malformed", "last_event_id": "...", "reconcile_via": ["GET /tasks/{id}", ...]}` | 续传失效控制事件：世代失配或所求位置已被淘汰，调用方须走 REST 对账而非静默错位续推；随后转实时推送 |
| `finalized` | `{"chapters": [{"chapter_id", "text", "summary"}], "citation_warnings": [...]}` | 定稿全文（原始角标）；发出后流结束 |
| `error` | `{"message": "..."}` | 任务失败；发出后流结束 |

`product` 事件的 `kind` 与载荷：

| `kind` | 载荷 | 产出时机 |
|---|---|---|
| `outline_ready` | `{"kind": "outline_ready", "outline": [{id, title, subsections, chapter_type, planned_summary, points: [{id, text, hypotheses: [...]}]}]}` | 目录生成完成（含假说）；首跑框架阶段产出时推送，恢复续跑不重发（图不重跑已完成节点），回滚到框架前重放会重发 |
| `materials_ready` | `{"kind": "materials_ready", "chapter_id": "ch1", "materials": [Material...]}` | 该章素材落库；每章素材集合增长时推送，载荷为该章当前整块素材 |
| `chapter_ready` | `{"kind": "chapter_ready", "chapter_id": "ch1", "draft": {chapter_id, text, summary, self_check}}` | 该章正文写完；草稿文本变化时推送，载荷为该章整块草稿 |
| `review_pack_ready` | `{"kind": "review_pack_ready", "iteration_round": n, "chapter_ids": [...], "chapter_total": n, "chapter_completed": n, "material_count": n, "citation_warning_count": n, "review_warning_count": n, "revision_round_count": n, "pack_version": "..."}` | 停审阅门时与 `review_required` 同发（先产物后信号）；只推摘要 + `pack_version` 轮次指纹，绝不含章正文或素材全文；丢了靠 `GET /tasks/{id}/review` 重取全文。`pack_version` 与 `GET /review` 同源（同检查点同指纹） |

`product` 事件载荷与 `GET /tasks/{id}/products` 章级快照逐字段同构——丢帧后按 REST 取回同等内容；审阅包摘要丢了按 `GET /tasks/{id}/review` 取回全文。

`content_delta` 逐字流语义：

- `attempt`：退化重试开启新一轮流式；同一 `(chapter_id, mode)` 下更高 `attempt` 意味着调用方须丢弃之前所有 delta、从零重建；`sequence` 在每个 attempt 内独立从 0 单调递增。
- 合并粒度（时间窗口 / 字符数）可配（环境变量 `WRITER_DELTA_FLUSH_CHARS` / `WRITER_DELTA_FLUSH_MS`），合帧在工作线程侧完成后才过线程边界，避免逐 token 跨线程调度。
- `content_delta` 非持久化、不入 `graph_events` 可视化通道；`chapter_ready` 整块是持久锚，丢了逐字帧靠该整块对账。

### 3.2 可视化通道 `GET /graph_events`

全局流（覆盖所有任务），**永不主动关闭**，由客户端断开；用于执行拓扑可视化与运维观测，非业务必需。
传输层与业务通道一致（`sse-starlette` + `{epoch}-{seq}` 传输 id + `Last-Event-ID` 续传 + keepalive + 断线检测）；12 个事件类型与元数据专用性质不变。

查询参数（均可选，组合过滤）：

| 参数 | 说明 |
|---|---|
| `thread_id` | 只订阅某任务 |
| `session_id` | 只订阅某会话（即调用方创建任务时透传的值） |
| `types` | 逗号分隔的事件类型白名单；含非法值 → 400 |

每帧 `id` 行为传输 id `{epoch}-{seq}`（供 `Last-Event-ID` 续传）；`data` 为完整事件信封，其 `event_id` 为拓扑 uuid（`parent_id` 据此拼接，与传输 id 相互独立）：

```json
{
  "event_id": "拓扑 uuid",
  "trace_id": "任务的 execution_trace_id",
  "session_id": "调用方透传值",
  "thread_id": "...",
  "parent_id": "父事件 event_id 或 null",
  "ts": "UTC ISO8601",
  "type": "node_start",
  "unit": "运行单元名或 graph",
  "payload": { }
}
```

12 个事件类型：`node_start`、`node_end`、`node_error`、`gate_blocked`、`gate_resumed`、`branch_taken`、`loop_iteration`、`subagent_start`、`subagent_end`、`state_snapshot`、`llm_config_used`、`progress`。
`parent_id` 用于拼接执行拓扑树。
`state_snapshot` 只含元数据（状态、轮次、章节与素材计数等），绝不含正文。

### 3.3 背压可观测 `GET /tasks/{thread_id}/stream/stats` 与 `GET /graph_events/stats`

只读、轻量，返回 SSE 通道的背压健康指标：

```json
{ "thread_id": "...", "subscriber_count": n, "dropped": n, "epoch": "..." }
```

| 字段 | 说明 |
|---|---|
| `subscriber_count` | 该通道当前在线订阅者数 |
| `dropped` | 累计丢弃事件数（历史缓冲淘汰 + 慢消费者队列丢弃） |
| `epoch` | 通道世代 id，供客户端核对续传是否同世代 |

业务通道 stats 须带真实 `thread_id`（未知任务 → 404）；可视化通道 stats 为全局聚合、不带 `thread_id` 字段。可视化通道事件皆元数据信封（不可丢级），`dropped` 主要反映历史缓冲淘汰。

## 4. Java 端对接示例

以下示例基于 Spring WebFlux `WebClient`（也可换用 OkHttp / RestClient，协议是普通 HTTP + SSE）。

### 4.1 创建任务

```java
record CreateTaskRequest(String userIntent, String userIdentity, String sessionId) {}
record CreateTaskResponse(String threadId, String executionTraceId) {}

WebClient client = WebClient.create("http://writer-host:8000");

// 注意：服务端字段为 snake_case，需配置 Jackson SNAKE_CASE 命名策略或用 @JsonProperty。
CreateTaskResponse task = client.post()
    .uri("/tasks")
    .contentType(MediaType.APPLICATION_JSON)
    .bodyValue(Map.of(
        "user_intent", "写一篇论证国产数据库替代可行性的行业白皮书",
        "user_identity", "u-1001",
        "session_id", "sess-abc"))
    .retrieve()
    .bodyToMono(CreateTaskResponse.class)
    .block();
```

### 4.2 订阅业务 SSE

```java
ParameterizedTypeReference<ServerSentEvent<String>> sseType =
    new ParameterizedTypeReference<>() {};

client.get()
    .uri("/tasks/{id}/stream", task.threadId())
    .accept(MediaType.TEXT_EVENT_STREAM)
    .retrieve()
    .bodyToFlux(sseType)
    .doOnNext(event -> {
        String type = event.event();   // status / review_required / finalized / error
        String json = event.data();    // 完整事件 JSON，用 Jackson 解析 data 字段
        switch (type) {
            case "review_required" -> submitReview(task.threadId());
            case "finalized" -> saveArticle(json);
            case "error" -> handleFailure(json);
            default -> log.info("status: {}", json);
        }
    })
    // finalized / error 后服务端正常收流；中途断线带 Last-Event-ID 重连续传（只补该 id 之后的事件，不全文回放）；世代失配时收 reconcile_required 控制事件后转实时，并按载荷走 REST 对账。
    .blockLast();
```

### 4.3 提交审阅

```java
// 修订一轮
client.post()
    .uri("/tasks/{id}/review", task.threadId())
    .contentType(MediaType.APPLICATION_JSON)
    .bodyValue(Map.of("action", "revise",
                      "feedback", "引言口吻克制些；第二章补充行业数据佐证"))
    .retrieve()
    .toBodilessEntity()
    .block();   // 409 表示任务未停在中断点或正在运行，稍后重试

// 定稿
client.post()
    .uri("/tasks/{id}/review", task.threadId())
    .contentType(MediaType.APPLICATION_JSON)
    .bodyValue(Map.of("action", "finalize"))
    .retrieve()
    .toBodilessEntity()
    .block();
```

### 4.4 获取最终交付

```java
String delivery = client.get()
    .uri(uri -> uri.path("/tasks/{id}/bibliography")
                   .queryParam("format", "gbt7714")
                   .build(task.threadId()))
    .retrieve()
    .bodyToMono(String.class)
    .block();
```

### 4.5 对接要点

- 服务端 JSON 字段均为 snake_case；Java DTO 建议全局配置 `PropertyNamingStrategies.SNAKE_CASE`。
- `session_id` 由 Java 端生成（如每用户会话一个 UUID）；用 `/graph_events?session_id=` 可按会话过滤可视化事件。
- 提交审阅前可先 `GET /tasks/{id}` 确认 `awaiting_review=true && running=false`，或直接以 409 作为并发控制信号重试。
- 服务重启后任务登记在内存中丢失但检查点仍在 Postgres：对旧任务先调 `POST /tasks/{id}/resume` 再继续操作。
- 最终产物两种取法：`finalized` 事件（原始素材 id 角标）或 `GET .../bibliography`（重编号正文 + 书目，推荐）。
