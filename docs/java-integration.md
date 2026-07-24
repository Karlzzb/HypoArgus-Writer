# HypoArgus-Writer Java 对接说明

这份文档是给 Java 服务端看的简版对接说明。
它讲最常用的调用顺序、状态判断、对外数据结构、过程内容获取、打字机效果获取和最终结果获取。
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

## 3. 暴露给调用端的数据结构

这一节是调用端视角的完整数据模型。
系统内部的运行 state 不直接暴露，对外暴露的是它在各接口上的投影。
好消息是：所有接口和 SSE 事件共用同一套类型，学一遍就能读懂全部载荷。

### 3.1 任务 State 总图

把所有接口能拿到的内容合在一起，一个任务在调用端眼里就是下面这棵树。

```
任务 (thread_id)
├── 运行元信息：status / iteration_round / awaiting_review / running / mock
├── chapters[]                     ← 章级快照（products 接口的主体）
│   ├── 章骨架：chapter_id / title / subsections / chapter_type / planned_summary
│   ├── points[]                   ← 论点
│   │   └── hypotheses[]           ← 假说
│   ├── materials[]                ← 该章素材（Material）
│   └── draft                      ← 该章草稿（ChapterDraft，未写完为 null）
├── citation_warnings[]            ← 引文未决警告（审阅包）
├── review_warnings[]              ← 篇级 warn 提示（审阅包）
├── revision_ledger[]              ← 修订台账（RevisionRound，审阅包）
└── 最终交付：重编号正文 + bibliography[]（bibliography 接口）
```

层级关系记一句话就够：一个任务有多章，一章有多个论点，一个论点有多条假说，每条假说回链多条素材，每章最终落一份草稿。

### 3.2 任务状态枚举

`status` 字段在所有接口和 `status` 事件里都是同一个枚举。

| 值 | 含义 |
|---|---|
| `IDLE` | 刚创建，尚未开跑。 |
| `FRAMEWORK_BUILDING` | 正在生成目录和假说。 |
| `REFERENCE_FETCHING` | 正在检索素材。 |
| `ARTICLE_WRITING` | 正在写章节正文。 |
| `CITATION_CHECKING` | 正在做篇级终审。 |
| `AWAIT_USER_REVIEW` | 停在人工审阅点，等你提交审阅。 |
| `FINISHED` | 已定稿。 |
| `ERROR_FAILED` | 任务失败。 |

### 3.3 章骨架与假说：ChapterSpec / ArgumentPoint / Hypothesis

大纲的每一章是一个 `ChapterSpec`。
它出现在 `products` 的 `chapters[]`、`review` 的 `outline[]` 和 `product` 事件的 `outline_ready` 载荷里。

| 字段 | 类型 | 说明 |
|---|---|---|
| `chapter_id` | string | 章标识，形如 `ch1`；`outline_ready` 事件里记为 `id`，同源同值。 |
| `title` | string | 章标题。 |
| `subsections` | string[] | 三级标题列表。 |
| `chapter_type` | string \| null | 模板骨架章标题原文；自由结构模式为 `null`。 |
| `planned_summary` | string | 框架阶段预判的本章一句话概要。 |
| `points` | ArgumentPoint[] | 本章论点列表。 |

`ArgumentPoint` 是章内的一个中心主张。

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string | 论点标识。 |
| `text` | string | 论点内容。 |
| `hypotheses` | Hypothesis[] | 从论点派生的假说列表。 |

`Hypothesis` 是可检索验证的具体命题，素材靠它回链。

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string | 假说标识，素材的 `hypothesis_id` 指向它。 |
| `text` | string | 假说内容。 |
| `refute_condition` | string | 证伪条件。 |
| `angle` | string | 假说角度，六值之一：假设 / 失效模式 / 边界条件 / 竞争解释 / 预言 / 反事实。 |

### 3.4 素材：Material

`Material` 是全系统统一的素材形状。
它出现在 `products` 的 `chapters[].materials`、`review` 的 `citation_library`、`bibliography` 的 `bibliography[].material` 和 `materials_ready` 事件里，四处字段完全一致。

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string | 稳定不透明 id，形态固定为 `m_<26位CrockfordBase32>`；正文角标嵌的就是它，跨接口关联也靠它。 |
| `hypothesis_id` | string | 回链的假说 id。 |
| `chapter_id` | string | 所属章 id。 |
| `source` | string | 来源名称。 |
| `url` | string \| null | 来源链接；仅联网来源必带，知识库与结构化来源可为 `null`。 |
| `source_kind` | string | 来源通道三值：`web` / `knowledge_base` / `structured_data`。 |
| `source_ref` | object \| null | 真实来源定位；web 通常含 `url`，知识库通常含 `knowledge_id`/`file_id`/`chunk_id`，结构化数据通常含 `scenario_key`/`dataset_id`/`query_execution_id`。 |
| `excerpt` | string | 证据摘录。 |
| `relevance_score` | number | 相关性打分。 |
| `verdict` | string | 佐证强度三值：`pass`（强支撑）/ `inconclusive`（弱佐证，仅作背景提示）/ `fail`（反例或不可用，供审计）。 |

两条铁律。

1. `id` 只承担引用身份，不承载来源定位明文；真实定位一律读 `source_ref`。
2. `verdict` 是三值，消费方必须按三值处理，不能当布尔用。

Java DTO 建议把 `source_ref` 建成 `Map<String, Object>`，因为三条来源通道的定位字段不同。

### 3.5 章草稿：ChapterDraft / SelfCheck

`ChapterDraft` 是单章正文的完整形状。
它出现在 `products` 的 `chapters[].draft` 和 `chapter_ready` 事件的 `draft` 载荷里，两处严格同构。

| 字段 | 类型 | 说明 |
|---|---|---|
| `chapter_id` | string | 所属章 id。 |
| `text` | string | 章正文，含原位素材 id 角标（未重编号）。 |
| `summary` | string | 本章一句话摘要。 |
| `self_check` | SelfCheck | 单章自检结果。 |

`SelfCheck` 只有两个字段。

| 字段 | 类型 | 说明 |
|---|---|---|
| `citations_ok` | boolean | 引用自检是否通过。 |
| `issues` | string[] | 自检发现的问题列表。 |

`review` 和 `finalized` 事件里的 `chapters[]` 是它的精简投影，只含 `chapter_id` / `text` / `summary`。

### 3.6 修订台账：RevisionRound / RevisionDirective

`revision_ledger` 记录每一轮人工修订，出现在 `review` 审阅包里。

`RevisionRound`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `round_no` | int | 轮次号。 |
| `raw_feedback` | string | 你当轮提交的原始意见。 |
| `directives` | RevisionDirective[] | 系统解析出的结构化修订指令。 |
| `digest` | string \| null | 更早轮次压缩后的一句话摘要；最近轮次保留原文时为 `null`。 |

`RevisionDirective`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `target_chapter_id` | string | 目标章 id。 |
| `type` | string | 二值：`rewrite_only`（纯改写）/ `evidence_augmented`（补充佐证）。 |
| `instruction` | string | 具体修订指令。 |

`review_required` 事件里的 `pending_confirmation.directives` 用的也是这个形状。

### 3.7 最终书目条目

`bibliography` 接口返回重编号正文和书目条目数组。

| 字段 | 类型 | 说明 |
|---|---|---|
| `index` | int | 服务端生成的最终展示编号，Java 端不需要、也不应该重新编号。 |
| `material_id` | string | 兼容旧消费者，恒等于 `material.id`。 |
| `material` | Material | 完整素材，形状见 3.4。 |
| `text` | string | 按所选格式渲染好的书目文本。 |

### 3.8 类型出现位置速查

| 类型 | REST 出现位置 | SSE 出现位置 |
|---|---|---|
| 状态枚举 | `GET /tasks/{id}`、`products`、`checkpoints` | `status` 事件 |
| ChapterSpec | `products.chapters[]`、`review.outline[]` | `outline_ready` |
| ArgumentPoint / Hypothesis | 随 ChapterSpec 内嵌 | 随 `outline_ready` 内嵌 |
| Material | `products.chapters[].materials`、`review.citation_library`、`bibliography[].material` | `materials_ready` |
| ChapterDraft | `products.chapters[].draft` | `chapter_ready` |
| RevisionRound | `review.revision_ledger` | 无 |
| 书目条目 | `bibliography` | 无 |

所有 JSON 字段都是 snake_case。
Java DTO 建议全局配置 Jackson 的 `PropertyNamingStrategies.SNAKE_CASE`，每种类型建一个 record，跨接口直接复用。

## 4. 什么时候拿过程中的内容

### 4.1 运行中产物

接口：`GET /tasks/{thread_id}/products`

用途：拿运行过程里的内容快照。

这个接口适合做两件事。

1. SSE 丢了 `product` 事件时，用它补数据。
2. 你想在任务没结束时，查看已经生成到哪一章、哪些素材已经回来、哪些草稿已经写出。

返回内容就是 3.1 总图里的章级快照：

| 内容 | 说明 |
|---|---|
| `chapters` | 各章的快照，含骨架、论点、假说（形状见 3.3）。 |
| `chapters[].materials` | 每章已收集到的素材（形状见 3.4）。 |
| `chapters[].draft` | 该章草稿正文（形状见 3.5），未写完为 `null`。 |

这个接口是"过程数据"的主入口。
它不会像审阅包那样要求任务必须停下来。
未完成的部分由字段值表达：章未检索时 `materials` 为空，未写完时 `draft` 为 `null`，尚无大纲时 `chapters` 为空列表。

### 4.2 人工审阅包

接口：`GET /tasks/{thread_id}/review`

用途：拿停在审阅点时的完整内容包。

这个接口只有在任务已经停在人工审阅点时才能调。
如果任务还在跑，接口会返回 409。

这个包里会一次性给你：

| 内容 | 说明 |
|---|---|
| `outline` | 当前大纲（ChapterSpec 数组，见 3.3）。 |
| `chapters` | 各章正文（`chapter_id` / `text` / `summary`，见 3.5）。 |
| `citation_warnings` | 引文警告（字符串数组）。 |
| `review_warnings` | 篇级提示（字符串数组）。 |
| `revision_ledger` | 修订记录（RevisionRound 数组，见 3.6）。 |
| `citation_library` | 引文库素材（Material 数组，见 3.4）。 |

如果你要让人审稿、看全文、提修改意见，就用这个接口。

## 5. 什么时候拿打字机效果

### 5.1 SSE 里的逐字输出

接口：`GET /tasks/{thread_id}/stream`

打字机效果不在单独的 REST 接口里。
它通过 SSE 的 `content_delta` 事件推送。

Java 端收到这个事件后，把同一章、同一轮、同一模式下的 `delta` 拼起来，就能还原实时正文。

你只需要记住三点。

1. `content_delta` 是逐字增量。
2. 它可能会分多次到达。
3. 真正稳定可落地的结果，还是要以 `chapter_ready` 或最终 `bibliography` 为准。

### 5.2 过程内容和逐字流的区别

`content_delta` 适合做实时展示。
`products` 适合做过程对账。
`review` 适合做人工审阅。
`bibliography` 适合做最终交付。

## 6. 什么时候提交审阅

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

## 7. 什么时候拿最终结果

接口：`GET /tasks/{thread_id}/bibliography`

用途：拿最终正文和书目。

这是推荐的最终交付接口。

原因很简单。
它会返回重编号后的正文和配套书目（条目形状见 3.7），适合直接落库、导出或展示。

如果你只看 SSE 里的 `finalized` 事件，也能拿到最终正文。
但它保留的是原始角标。
需要正式交付时，还是建议用 `bibliography` 接口。

## 8. 一套最常用的 Java 调用顺序

### 8.1 标准流程

1. `POST /tasks` 创建任务。
2. `GET /tasks/{thread_id}/stream` 订阅进度和逐字输出。
3. 需要看运行内容时，调 `GET /tasks/{thread_id}/products`。
4. 任务停在审阅点后，调 `GET /tasks/{thread_id}/review`。
5. 提交 `revise`、`confirm` 或 `finalize`。
6. 最后调 `GET /tasks/{thread_id}/bibliography` 拿交付结果。

### 8.2 Java 侧建议

1. `session_id` 建议由 Java 端自己生成并保存。
2. `thread_id` 要作为任务主键一直传下去。
3. SSE 断线后要带 `Last-Event-ID` 重连。
4. 过程数据丢了就用 `products` 或 `review` 补。
5. 最终交付以 `bibliography` 为准。
6. DTO 按第 3 节的类型各建一个 record，跨接口复用，不要按接口各建一套。

## 9. 最简结论

如果你只记三件事，就记这三件事。

1. 看进度和打字机效果，用 `GET /tasks/{thread_id}/stream`。
2. 看过程中的完整内容，用 `GET /tasks/{thread_id}/products`。
3. 看最终交付结果，用 `GET /tasks/{thread_id}/bibliography`。

审阅点到来后，再用 `GET /tasks/{thread_id}/review` 和 `POST /tasks/{thread_id}/review` 完成人工接入。
