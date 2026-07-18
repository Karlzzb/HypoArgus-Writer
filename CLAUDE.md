# HypoArgus-Writer

纯 LangGraph 单一技术栈的工业级结构化写作后端服务。
项目唯一有效 PRD 见根目录 `PRD.md`；领域术语以 `CONTEXT.md` 词汇表为准。
全部文档与注释使用平实中文术语。

## Agent skills

### Issue tracker

Issues 存于本仓库的 GitHub Issues（`Karlzzb/HypoArgus-Writer`），用 `gh` CLI 操作。See `docs/agents/issue-tracker.md`.

### Triage labels

使用五个规范标签的默认命名（`needs-triage`、`needs-info`、`ready-for-agent`、`ready-for-human`、`wontfix`）。See `docs/agents/triage-labels.md`.

### Domain docs

单一上下文布局——仓库根 `CONTEXT.md` 词汇表 + `docs/adr/`。See `docs/agents/domain.md`.
