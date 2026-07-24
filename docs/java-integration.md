# HypoArgus-Writer Java 对接说明

这份文档是给 Java 服务端看的简版对接说明。
它只讲最常用的调用顺序、状态判断、过程内容获取、打字机效果获取和最终结果获取。
底层接口仍然是 `docs/api.md` 中定义的 REST + SSE。

## 1. 先看整体流程

Java 端一般按下面的顺序调用。

1. `POST /tasks` 创建写作任务。
2. `GET /tasks/{thread_id}/stream` 订阅任务进度。
3. 任务停在人工审阅点后，调用 `GET /tasks/{thread_id}/review` 取完整审阅包。
4. 根据审阅结果，调用 `POST /tasks/{thread_id}/review` 提交 `revise`、`confirm` 或 `finalize`。
5. 需要看运行中的内容时，调用 `GET /tasks/{thread_id}/products`。
6. 需要看最终可交付正文和书目时，调用 `GET /tasks/{thread_id}/bibliography`。

## 2. Java 什么时候调哪个接口

### 2.1 创建任务

接口：`POST /tasks`

用途：开始一次新的写作任务。

常用请求字段：

| 字段 | 说明 |
|---|---|
| `user_intent` | 写作需求，必填。 |
| `user_identity` | 用户标识，可选。 |
| `session_id` | 会话标识，可选，只透传。 |
| `mock` | 是否使用 mock 任务，可选。 |

返回值里最重要的是 `thread_id`。
后续所有接口都靠它定位任务。

### 2.2 订阅任务进度

接口：`GET /tasks/{thread_id}/stream`

用途：看任务现在走到哪一步了。

Java 端一般在创建任务后立刻建立这个 SSE 长连接。

这个流里最常见的事件是：

| 事件 | 含义 |
|---|---|
| `status` | 任务状态变化。 |
| `product` | 产物块更新，例如目录、素材、章节草稿。 |
| `content_delta` | 打字机效果，正文逐字输出。 |
| `review_required` | 任务停在人工审阅点。 |
| `finalized` | 最终完成。 |
| `error` | 任务失败。 |

### 2.3 获取任务当前状态

接口：`GET /tasks/{thread_id}`

用途：只看任务现在是否还在跑、是否停在审阅点。

重点字段：

| 字段 | 说明 |
|---|---|
| `status` | 当前阶段。 |
| `awaiting_review` | 是否已经停在人工审阅点。 |
| `running` | 是否还在运行中。 |
| `iteration_round` | 当前第几轮。 |

Java 端通常在发审阅前先查一次这个接口。
如果 `awaiting_review=true` 且 `running=false`，再提交审阅最稳妥。

## 3. 什么时候拿过程中的内容

### 3.1 运行中产物

接口：`GET /tasks/{thread_id}/products`

用途：拿运行过程里的内容快照。

这个接口适合做两件事。

1. SSE 丢了 `product` 事件时，用它补数据。
2. 你想在任务没结束时，查看已经生成到哪一章、哪些素材已经回来、哪些草稿已经写出。

返回内容里一般会有：

| 内容 | 说明 |
|---|---|
| `chapters` | 各章的快照。 |
| `materials` | 每章已收集到的素材。 |
| `draft` | 当前草稿正文。 |

这个接口是“过程数据”的主入口。
它不会像审阅包那样要求任务必须停下来。

### 3.2 人工审阅包

接口：`GET /tasks/{thread_id}/review`

用途：拿停在审阅点时的完整内容包。

这个接口只有在任务已经停在人工审阅点时才能调。
如果任务还在跑，接口会返回 409。

这个包里会一次性给你：

| 内容 | 说明 |
|---|---|
| `outline` | 当前大纲。 |
| `chapters` | 各章正文。 |
| `citation_warnings` | 引文警告。 |
| `review_warnings` | 篇级提示。 |
| `revision_ledger` | 修订记录。 |
| `citation_library` | 引文库素材。 |

如果你要让人审稿、看全文、提修改意见，就用这个接口。

## 4. 什么时候拿打字机效果

### 4.1 SSE 里的逐字输出

接口：`GET /tasks/{thread_id}/stream`

打字机效果不在单独的 REST 接口里。
它通过 SSE 的 `content_delta` 事件推送。

Java 端收到这个事件后，把同一章、同一轮、同一模式下的 `delta` 拼起来，就能还原实时正文。

你只需要记住三点。

1. `content_delta` 是逐字增量。
2. 它可能会分多次到达。
3. 真正稳定可落地的结果，还是要以 `chapter_ready` 或最终 `bibliography` 为准。

### 4.2 过程内容和逐字流的区别

`content_delta` 适合做实时展示。
`products` 适合做过程对账。
`review` 适合做人工审阅。
`bibliography` 适合做最终交付。

## 5. 什么时候提交审阅

接口：`POST /tasks/{thread_id}/review`

用途：在任务停住后继续往下走。

常见动作有三个。

| 动作 | 什么时候用 |
|---|---|
| `revise` | 让系统按你的意见再改一轮。 |
| `confirm` | 确认大范围修订清单后继续。 |
| `finalize` | 直接定稿结束。 |

`revise` 时要带 `feedback`。

示例：

```json
{ "action": "revise", "feedback": "引言口吻克制些；第二章补充行业数据佐证" }
```

如果任务还没停在审阅点，接口会返回 409。

## 6. 什么时候拿最终结果

接口：`GET /tasks/{thread_id}/bibliography`

用途：拿最终正文和书目。

这是推荐的最终交付接口。

原因很简单。
它会返回重编号后的正文和配套书目，适合直接落库、导出或展示。

如果你只看 SSE 里的 `finalized` 事件，也能拿到最终正文。
但它保留的是原始角标。
需要正式交付时，还是建议用 `bibliography` 接口。

## 7. 一套最常用的 Java 调用顺序

### 7.1 标准流程

1. `POST /tasks` 创建任务。
2. `GET /tasks/{thread_id}/stream` 订阅进度和逐字输出。
3. 需要看运行内容时，调 `GET /tasks/{thread_id}/products`。
4. 任务停在审阅点后，调 `GET /tasks/{thread_id}/review`。
5. 提交 `revise`、`confirm` 或 `finalize`。
6. 最后调 `GET /tasks/{thread_id}/bibliography` 拿交付结果。

### 7.2 Java 侧建议

1. `session_id` 建议由 Java 端自己生成并保存。
2. `thread_id` 要作为任务主键一直传下去。
3. SSE 断线后要带 `Last-Event-ID` 重连。
4. 过程数据丢了就用 `products` 或 `review` 补。
5. 最终交付以 `bibliography` 为准。

## 8. 最简结论

如果你只记三件事，就记这三件事。

1. 看进度和打字机效果，用 `GET /tasks/{thread_id}/stream`。
2. 看过程中的完整内容，用 `GET /tasks/{thread_id}/products`。
3. 看最终交付结果，用 `GET /tasks/{thread_id}/bibliography`。

审阅点到来后，再用 `GET /tasks/{thread_id}/review` 和 `POST /tasks/{thread_id}/review` 完成人工接入。
