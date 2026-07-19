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

## 写作子智能体（rewriter_loop）

- **风格指南（Style Guide）** — 随 rewriter_loop 包携带的公文写作规范文件，代码级单一事实源，路径不可配。
  散文部分注入写作提示词，内嵌 YAML 块（ssot-config）供校验器解析词表与规则，同一文件双消费、零漂移。
- **确定性风格校验（Style Lint）** — 对单章正文执行注册表内全部规则的纯函数校验，不依赖 LLM。
  规则涵盖公文措辞、章型结构、意识形态串归位、素材角标与查臆造，违规折叠进 self_check.issues。
- **写作 LLM 缝（WriterLlmClient）** — rewriter_loop 包内的 LLM 协议缝，含 draft/revise/audit 三方法。
  编排层只依赖该协议，真实适配器与确定性 Fake 可互换，测试零网络零成本。
- **退化重试（Degradation Retry）** — 模型应答异常、解析失败、结构非法或正文为空时视为退化并重试至上限的兜底模型。
  重试耗尽后诚实返回（标 degraded）或重抛异常，不以脏数据蒙混。
- **LLM 自审（Audit）** — 用模型判断正文是否存在改写自素材原文却未挂角标的「派生未标」违规。
  空裁决合法不重试；重试耗尽降级为空裁决，永不阻断写作主链。
- **修一次（Fix-once）** — 校验与自审检出任一违规时恰好触发一次修订重写，修后不复检。
  self_check 折叠的是修前质检结论，修订产物的最终合规由全局终审兜底。

## 上下文装配

- **上下文装配（Context Assembly）** — 每次 LLM 调用或子智能体任务包的输入由 context_assembler 从 State 现场装配的机制，禁止透传原始累积历史。
  统一入口 `assemble(state, unit)`，差异收敛于按运行单元注册的装配配方。
- **装配配方（Recipe）** — 一个运行单元的提取器组合，阈值与保留策略缺省全部取装配配置。
  差异只体现在配方，提取器可跨配方复用。
- **提取器（Extractor）** — State 加调用点局部参数到内容段列表的纯函数，可跨配方复用。
  禁止读取 State 之外的全局可变状态；定位类参数（如 chapter_id）缺失时返回空段而不抛错。
- **内容段（Segment）** — 装配产出的最小单位，形如 段名 + 文本，供节点按名取用构造 prompt 或任务包字段。
- **修订台账（Revision Ledger）** — 用户历轮意见与解析出的修订指令的全量持久化记录，装配时按需注入，保证多轮迭代不失忆。
- **摘要链（Summary Chain）** — 已完成各章摘要按顺序拼接成的前文链，供当前章写作承接；超阈值时压缩，未超时原样拼接，首章为空。
- **保留策略（Retention Policy）** — 装配时对过长上下文的压缩规则：摘要链过长做「摘要的摘要」；修订台账保最近 K 轮原文加更早轮次一句话摘要。
  压缩只在超阈值时发生，阈值可配置。
