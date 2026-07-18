# HypoArgus-Writer 领域词汇表

本文件只做术语表，不含实现细节。

## 论证结构

- **章节（Chapter）** — 文章的二级标题单元，写作与检索的基本执行粒度。
  一个章节包含 1..n 个论点。
  章节严格串行写作，承接上一章节摘要。
- **论点（Argument Point）** — 章节内的一个中心主张，章节存在的理由之一。
  一个论点派生 1..N 条假说。
- **假说（Hypothesis）** — 从论点派生的可证伪、可检索验证的具体命题。
  每条假说必须声明其证伪条件。
  假说是 search_agent 检索任务的直接驱动源；被筛掉（证据不可检索）的假说不进入 State。
  生成机制继承六角度框架：假设 / 失效模式 / 边界条件 / 竞争解释 / 预言 / 反事实。

层级关系：章节 1—n 论点，论点 1—N 假说。
各层数量上限均为可配置项，非固定值。

## 运行单元

- **运行单元（Runtime Unit）** — 可独立配置 LLM 参数（model / base_url / api_key）的最小执行体。
  共 7 个：5 个 LangGraph 主节点（framework_orchestrator、reference_orchestrator、writing_orchestrator、citation_validator、human_review_gate）+ 2 个业务子智能体（search_agent、rewriter_loop）。
  未单独配置的单元回落到全局缺省 LLM 配置。

## 修订

- **修订指令（Revision Directive）** — human_review_gate 将用户一次提交的自然语言修改意见解析出的结构化最小修订单位，形如 {目标章节, 类型}。
  一次用户意见可拆解为多条修订指令，混合两种类型，在同一轮迭代内各自执行。
- **纯改写（Rewrite-only）** — 修订指令类型之一：不需要新证据、仅调整文字表达，走 rewriter_loop 分支。
- **补充佐证（Evidence-augmented）** — 修订指令类型之一：需要新素材支撑，先走 search_agent 增量检索再改写。
