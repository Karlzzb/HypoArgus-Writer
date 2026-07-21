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
| 人工审阅 | 任务写完一轮后停在人工中断点（`awaiting_review=true`），调用方提交 `finalize`（定稿）或 `revise`（携修订意见再迭代一轮），可无限迭代 |
| 检查点 | 每个关键步骤持久化到 Postgres；支持崩溃恢复与回滚到任意历史检查点 |

### 1.2 典型调用时序

```
Java 端                                  HypoArgus-Writer
  │ POST /tasks                            │
  │──────────────────────────────────────▶│ 201 {thread_id, execution_trace_id}
  │ GET /tasks/{id}/stream  (SSE 长连接)   │
  │──────────────────────────────────────▶│
  │        ◀── event: status  (多次，各阶段推进)
  │        ◀── event: review_required     （任务停在人工中断点）
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

```json
{ "user_intent": "写一篇论证国产数据库替代可行性的行业白皮书", "user_identity": "u-1001", "session_id": "sess-abc" }
```

响应体：

```json
{ "thread_id": "9f3c...32位hex", "execution_trace_id": "1d2e...32位hex" }
```

### 2.2 查询任务状态

`GET /tasks/{thread_id}` → `200 OK`

```json
{
  "thread_id": "9f3c...",
  "status": "AWAIT_USER_REVIEW",
  "iteration_round": 1,
  "awaiting_review": true,
  "running": false
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `status` | string | 见 1.1 状态枚举 |
| `iteration_round` | int | 当前迭代轮次，从 0 起 |
| `awaiting_review` | bool | 是否停在人工中断点（true 时才能提交审阅） |
| `running` | bool | 是否有图运行正在进行 |

错误：任务不存在 → 404。

### 2.3 提交人工审阅

`POST /tasks/{thread_id}/review` → `202 Accepted`

从人工中断点恢复运行。
仅当 `awaiting_review=true` 且 `running=false` 时可调，否则 409。

请求体：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `action` | string | 是 | `"finalize"`（定稿）或 `"revise"`（修订） |
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
    { "index": 1, "material_id": "m-ch1-p1-h1", "text": "[1] 来源[EB/OL]. url." }
  ]
}
```

错误：格式未注册 → 400；尚无章节正文（框架 / 检索阶段）→ 409。

说明：`finalized` SSE 事件中的正文保留原始素材 id 角标；本接口返回按出现顺序重编号后的正文与配套书目，是推荐的最终产物获取方式。

### 2.8 独立检索

`POST /retrieval` → `200 OK`

阻塞式独立检索：提交一章假说列表，同步等待检索完成并返回素材与诊断，不启动写作任务。
与写作主流程使用同一套任务包/结果契约与同一 search_agent 实例，两处调用行为一致。

请求体：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `chapter_id` | string | 是 | 章节标识，不允许空白，进事件上下文与素材 id |
| `hypotheses` | array | 是 | 假说列表，不允许为空；条目为 `{id, text, refute_condition}`，`id` 与 `text` 不允许空白，`refute_condition` 非空白时驱动反向检索 |
| `genre` | string | 否 | 品类（检索范围提示），缺省 `""` |
| `existing_materials_digest` | string | 否 | 既有素材摘要（供引擎规避重复素材），缺省 `""` |
| `session_id` | string | 否 | 调用方会话标识，缺省 `""`，进度事件按其入信封供 `/graph_events` 过滤订阅 |

```json
{
  "chapter_id": "ch1",
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
      "id": "m-ch1-h1-cit-1",
      "hypothesis_id": "h1",
      "source": "来源名称",
      "url": "https://example.com/evidence",
      "source_kind": "web",
      "excerpt": "证据摘录……",
      "relevance_score": 0.9,
      "verdict": "pass"
    }
  ],
  "diagnostics": { "total_elapsed_ms": 1234, "call_counts": { "web_search": 2 } }
}
```

| 字段 | 说明 |
|---|---|
| `materials[].hypothesis_id` | 素材回链的假说 id |
| `materials[].url` | 来源链接；仅联网来源必带，知识库与结构化来源可为 `null` |
| `materials[].source_kind` | 来源通道三值：`web` / `knowledge_base` / `structured_data` |
| `materials[].verdict` | `pass`（可作支撑证据）/ `fail`（反驳或不支撑，供筛选审计） |
| `diagnostics` | 本次检索的诊断摘要（计数与耗时），与 `subagent_end` 事件携带的诊断同源 |

进度事件：调用期间以请求中的 `session_id` 在 `/graph_events` 通道发布 `subagent_start` → `progress`（多条）→ `subagent_end` 事件链，`subagent_start` 为本次调用的根事件（`thread_id` 为空串）。

错误：请求体校验失败 → 422；检索任务违反引擎入参契约 → 422；检索通道 / LLM 配置缺失 → 503。

## 3. SSE 通道

两条通道，帧格式均为标准 SSE 三行：

```
id: <event_id>
event: <事件类型>
data: <JSON>

```

Java 端可用任意 SSE 客户端（如 OkHttp EventSource、Spring WebFlux）消费；`id` 行即 `event_id`。
建议客户端不设读超时（长连接），并做断线重连（重连后业务通道会先回放历史事件再实时推送）。

### 3.1 业务通道 `GET /tasks/{thread_id}/stream`

每任务一条流；订阅时先回放该任务历史事件再实时推送；任务定稿或失败后流**正常结束**。

`data` 统一结构：

```json
{ "event_id": "hex", "type": "...", "thread_id": "...", "ts": "UTC ISO8601", "data": { } }
```

| `type` | `data` 结构 | 说明 |
|---|---|---|
| `status` | `{"status": "...", "iteration_round": n, "node": "节点名"}` | 状态机推进 |
| `review_required` | `{"iteration_round": n, "chapter_ids": [...], "citation_warnings": [...], "error"?: "..."}` | 停在人工中断点，等待调用方提交审阅；`error` 仅在上次提交契约不符时出现 |
| `finalized` | `{"chapters": [{"chapter_id", "text", "summary"}], "citation_warnings": [...]}` | 定稿全文（原始角标）；发出后流结束 |
| `error` | `{"message": "..."}` | 任务失败；发出后流结束 |

### 3.2 可视化通道 `GET /graph_events`

全局流（覆盖所有任务），**永不主动关闭**，由客户端断开；用于执行拓扑可视化与运维观测，非业务必需。

查询参数（均可选，组合过滤）：

| 参数 | 说明 |
|---|---|
| `thread_id` | 只订阅某任务 |
| `session_id` | 只订阅某会话（即调用方创建任务时透传的值） |
| `types` | 逗号分隔的事件类型白名单；含非法值 → 400 |

每帧 `data` 为完整事件信封：

```json
{
  "event_id": "hex",
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
    // finalized / error 后服务端正常收流；中途断线可重连（会回放历史事件）。
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
