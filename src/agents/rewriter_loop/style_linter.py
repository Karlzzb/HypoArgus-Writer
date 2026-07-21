"""style_linter：单章正文的规则校验器（纯函数，不依赖主图与 LLM）。

入口签名：``lint(text, doc_type, doc_variant=None, *, style_guides_dir=None, ...) -> list[Violation]``。
规则按文种+变体两层加载（ADR-0005）：随包 ``style_guides/`` 目录每文种一份 md，
「通用公文.md」既是兑底文种指南也是跨文种通用规则层；其他文种做「通用 + 文种」
两层合并——列表类规则取并集，标量类规则文种覆盖通用。词表/禁用词/boilerplate
从各文件的 SSoT YAML 块解析加载，散文与 YAML 同文件双消费，避免指南与校验器漂移。
正文角标为本项目单方括号 ``[素材id]`` 语义，
解析复用 ``domain.citation_reconciler.MARKER_PATTERN``（唯一事实源，不另定义角标正则）。
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import yaml
from pydantic import BaseModel

from agents.contracts import HypothesisPayload, MaterialPayload
from domain.citation_reconciler import MARKER_PATTERN
from domain.doc_types import GENERIC_DOC_TYPE, tier_from_variant

# 随包携带的风格指南目录默认路径：不依赖工作目录，显式传参可覆盖（便于测试）。
STYLE_GUIDES_DIR = Path(__file__).parent / "style_guides"

_CHINESE_NUMERAL_PREFIX = re.compile(r"^[一二三四五六七八九十百]+[、､]\s*")
# markdown 表分隔行：起始 |、含 3+ 连字符、结尾 |（如 ``| ---- | ---- |``）。
_MD_TABLE_SEP = re.compile(r"^\s*\|[\s:|-]*-{3,}[\s:|-]*\|")
# 两个 CJK 字符之间的横向空白（PDF 抽取常见断词，如「新 时代」「职 业」）。
# 只压横向空白（空格/制表/全角空格），不吞换行——换行是标题/段落结构，不可丢。
_CJK_WS = re.compile(r"(?<=[一-鿿])[ \t　]+(?=[一-鿿])")


class Violation(BaseModel):
    """校验器单条违规。"""

    rule: str
    message: str
    severity: str = "error"


class Fact(BaseModel):
    """引用事实依据（行业代码/课程/学分/证书等），由调用方传入。

    与素材（``MaterialPayload``）严格分离：事实依据管「不得杜撰数值/名称」，
    素材管「论点可溯源到检索来源」。本切片保留其校验能力供后续切片调用。
    """

    type: Literal["industry_code", "course", "credit", "certificate", "other"]
    value: str


def normalize_cjk_ws(text: str) -> str:
    """抹去 CJK 字符之间的空白（PDF 抽取断词残留），反复替换直至稳定。

    语料常在 CJK 词中插入空格（「习近平新 时代中国特色社会主义思想」「1+X 证 书」）。
    这些中断的词汇不是风格违规，不应原样传给规则比对——
    此处归一化后，逐字串/术语/事实抽取的子串与正则匹配才不被断词假性破坏。
    仅处理 CJK 与 CJK 之间的空白，保留中英/中数之间的合法空格。
    """
    prev = text
    for _ in range(4):
        text = _CJK_WS.sub("", text)
        if text == prev:
            break
        prev = text
    return text


# 数值类事实的排版归一：全角数字/小数点/百分号/括号/冒号转半角（比对前统一口径）。
_FULLWIDTH_NUMERIC = str.maketrans("０１２３４５６７８９．％（）：", "0123456789.%():")
# 数值类事实类型：在位比对前须做排版归一（全角转半角 + 去空白），修排版差异漏判。
_NUMERIC_FACT_TYPES = frozenset({"credit", "industry_code"})


def normalize_numeric_text(value: str) -> str:
    """数值比对口径归一：全角数字/符号转半角，去除全部空白（含全角空格）。

    参考依据与正文常有排版差异（「７８学分」vs「78学分」、「78 学分」vs「78学分」），
    这些差异不是事实差异，比对前两侧同做此归一化，避免漏判/误伤。
    """
    return re.sub(r"\s+", "", value.translate(_FULLWIDTH_NUMERIC))


@dataclass(frozen=True)
class _LintContext:
    """一次 lint 调用中传给各规则的共享输入（正文已做 CJK 断词归一）。

    ``variant`` 是变体键解析结果（``tier_from_variant``），供变体分键规则
    （required_terms / forbidden_terms 等 dict[变体→词表]）取值；变体概念仅存活于
    人培方案文种内部——其他文种的合并配置不含变体分键规则，该值对其天然惰性。
    ``template`` 是章型标题：调用方传入 State 携带的章型（ADR-0005，骨架事实）
    时直接采用；未传（自由结构/旧链路）才回落从正文首个 ``## `` 标题反推。
    """

    text: str
    cfg: dict[str, Any]
    variant: str
    template: str | None
    domain: str | None
    references: list[Fact] | None
    materials: list[MaterialPayload] | None
    hypotheses: list[HypothesisPayload] | None


_Rule = Callable[[_LintContext], list[Violation]]

# 规则注册表：每条规则接收共享输入，返回违规列表。
_RULES: list[_Rule] = []


def register(rule: _Rule) -> _Rule:
    """把规则函数追加进注册表，供 ``lint`` 依序执行。"""
    _RULES.append(rule)
    return rule


def _resolve_guide_paths(
    doc_type: str, style_guides_dir: str | Path | None
) -> tuple[Path, Path | None]:
    """解析 (通用层文件, 文种层文件) 路径；通用层缺失即配置错误，立即抛出。

    文种层文件不存在返回 None——未落地专属指南的文种（汇报材料/学术论文占位）
    与兑底文种一样只走通用层，是 ADR-0005 已接受的架构后果，不是错误。
    """
    directory = Path(style_guides_dir) if style_guides_dir is not None else STYLE_GUIDES_DIR
    generic_path = directory / f"{GENERIC_DOC_TYPE}.md"
    if not generic_path.is_file():
        raise ValueError(f"style_guides 目录缺少通用层文件「{GENERIC_DOC_TYPE}.md」：{directory}")
    if doc_type == GENERIC_DOC_TYPE:
        return generic_path, None
    specific_path = directory / f"{doc_type}.md"
    return generic_path, specific_path if specific_path.is_file() else None


def _parse_config(path: Path) -> dict[str, Any]:
    """从单份指南文件解析 SSoT YAML 块（``<!-- ssot-config-begin ... -end -->`` 之间）。"""
    text = path.read_text(encoding="utf-8")
    m = re.search(r"ssot-config-begin\s*\n(.*?)\n\s*ssot-config-end", text, re.S)
    if not m:
        raise ValueError(f"风格指南未找到 ssot-config 块：{path}")
    config = yaml.safe_load(m.group(1))
    if not isinstance(config, dict):
        raise ValueError(f"风格指南的 ssot-config 块须为映射：{path}")
    return config


def _merge_layers(generic: dict[str, Any], specific: dict[str, Any]) -> dict[str, Any]:
    """两层合并语义（ADR-0005）：列表类取并集，标量类文种覆盖通用。

    - 两侧同为映射：逐键递归（嵌套结构如 ``word_count.chapter`` 内标量按键覆盖）。
    - 两侧同为列表：并集——通用层条目在前保序，文种层追加未重复条目
      （纯并集、无豁免/剔除机制，是通用层准入原则的合并侧保障）。
    - 其余（标量或异型）：文种层覆盖通用层。
    """
    merged = dict(generic)
    for key, value in specific.items():
        base = merged.get(key)
        if isinstance(base, dict) and isinstance(value, dict):
            merged[key] = _merge_layers(base, value)
        elif isinstance(base, list) and isinstance(value, list):
            merged[key] = base + [item for item in value if item not in base]
        else:
            merged[key] = value
    return merged


def load_config(
    doc_type: str, *, style_guides_dir: str | Path | None = None
) -> dict[str, Any]:
    """按文种加载合并后的 SSoT 配置：通用层必加载，文种层存在则两层合并。

    兑底文种（通用公文）与无专属指南文件的文种只得到通用层；
    其他文种得到「通用 + 文种」合并结果（列表并集、标量覆盖，见 ``_merge_layers``）。
    """
    generic_path, specific_path = _resolve_guide_paths(doc_type, style_guides_dir)
    config = _parse_config(generic_path)
    if specific_path is not None:
        config = _merge_layers(config, _parse_config(specific_path))
    return config


def _prose_of(path: Path) -> str:
    """读取单份指南文件的散文部分（``<!-- ssot-config-begin`` 之前）。"""
    text = path.read_text(encoding="utf-8")
    idx = text.find("<!-- ssot-config-begin")
    return text[:idx] if idx != -1 else text


def load_prose(doc_type: str, *, style_guides_dir: str | Path | None = None) -> str:
    """按文种读取散文部分（通用层散文 + 文种层散文），供注入写作提示词用。

    YAML 块是校验器的机器可读编码，散文已用人话描述同样规则，故只注入散文以
    节省上下文；YAML 仍由 ``load_config`` 独立解析，同一文件双消费、零漂移。
    两层散文按「通用在前、文种在后」拼接，与配置合并的层次方向一致。
    """
    generic_path, specific_path = _resolve_guide_paths(doc_type, style_guides_dir)
    prose = _prose_of(generic_path)
    if specific_path is not None:
        prose = prose.rstrip() + "\n\n" + _prose_of(specific_path)
    return prose


def detect_chapter_template(text: str, cfg: dict[str, Any] | None = None) -> str | None:
    """从正文首个 ``## `` 标题行解析章型标题。

    标题形如 ``## 五、职业面向`` → 去掉中文数字前缀 → ``职业面向``。
    返回标题本身（是否为 SSoT 已登记模板由各规则按需查 ``chapter_templates``）。
    """
    for line in text.splitlines():
        if line.startswith("## ") and not line.startswith("### "):
            title = line[3:].strip()
            title = _CHINESE_NUMERAL_PREFIX.sub("", title).strip()
            return title or None
    return None


def resolve_template(cfg: dict[str, Any], title: str | None) -> dict[str, Any] | None:
    """把章标题解析到 SSoT ``chapter_templates`` 中的模板（含别名匹配）。

    子风格用词分裂导致同名章型在不同层次有不同标题（如学制章：
    本科「学制学位」、高职「基本修业年限」）。模板可在 SSoT 登记
    ``aliases`` 把这些变体归并到同一模板，使 tier 专属规则（required_terms /
    forbidden_terms / forbidden_subsection_terms）对两种标题都生效。
    """
    if not title:
        return None
    templates = cfg.get("chapter_templates") or {}
    if title in templates:
        result: dict[str, Any] = templates[title]
        return result
    for tmpl in templates.values():
        if title in (tmpl.get("aliases") or []):
            resolved: dict[str, Any] = tmpl
            return resolved
    return None


def resolve_ideology_chapter(cfg: dict[str, Any], title: str | None) -> str | None:
    """把章标题归并到意识形态归位的规范章名（SSoT ``ideology.chapter_groups``）。

    语料章标题变体分裂（「培养目标及规格」/「培养目标与四大培养规格」←
    「培养目标与培养规格」；「课程设置及学时安排」/「课程体系与学分学时总览」/
    「课程架构」←「课程设置」；「教学进程总体安排」/「教学周期安排」←「教学进程」）。
    归位校验（``belongs_to`` / ``required_in``）用规范名匹配，使变体章同样受
    双向校验——否则变体章因标题不字面等于 belongs_to 而把归位串误判为 out_of_place。
    返回规范名；无映射则返回原标题（不会命中 belongs_to，等价于该章非归位章）。
    """
    if not title:
        return title
    groups = (cfg.get("ideology") or {}).get("chapter_groups") or {}
    if title in groups:
        return title
    for canonical, variants in groups.items():
        if title in (variants or []):
            return str(canonical)
    return title


@register
def _rule_oral_blacklist(ctx: _LintContext) -> list[Violation]:
    """禁讲客体/口语化表达与学术断言句式（黑名单来自 SSoT YAML）。

    逐字词条来自 ``oral_blacklist``（子串匹配）；句式词条来自
    ``oral_blacklist_patterns``（正则 ``search``，与编号黑名单同一模式机制），
    抓「XX是XX的必要条件」「XX正向预测XX」这类实证研究口吻。
    """
    out: list[Violation] = []
    for term in ctx.cfg.get("oral_blacklist", []):
        if term in ctx.text:
            out.append(
                Violation(
                    rule="oral_blacklist",
                    message=f"出现讲客体/口语化表达「{term}」，公文须用第三人称/无主语祈使。",
                )
            )
    for pat in ctx.cfg.get("oral_blacklist_patterns", []):
        m = re.search(pat, ctx.text)
        if m:
            out.append(
                Violation(
                    rule="oral_blacklist",
                    message=f"出现禁用断言句式「{m.group(0)}」（模式 {pat}），公文不用学术论文口吻。",
                )
            )
    return out


@register
def _rule_table_required(ctx: _LintContext) -> list[Violation]:
    """模板标记 table_required 的章须含 markdown 表。"""
    tmpl = resolve_template(ctx.cfg, ctx.template)
    if not tmpl or not tmpl.get("table_required"):
        return []
    has_table = any(_MD_TABLE_SEP.match(line) for line in ctx.text.splitlines())
    if not has_table:
        return [
            Violation(
                rule="table_missing",
                message=f"章型「{ctx.template}」须含 markdown 表，但未检出表格分隔行。",
            )
        ]
    return []


@register
def _rule_numbering(ctx: _LintContext) -> list[Violation]:
    """编号合规：禁用 SSoT ``numbering_blacklist_patterns`` 命中的起手式（如 ``1、``）。"""
    out: list[Violation] = []
    for line in ctx.text.splitlines():
        for pat in ctx.cfg.get("numbering_blacklist_patterns", []):
            if re.match(pat, line):
                out.append(
                    Violation(
                        rule="numbering",
                        message=f"行起手「{line.strip()}」命中禁用编号式 {pat}，公文章节用中文数字。",
                    )
                )
    return out


def _template_terms(tmpl: dict[str, Any], key: str, variant: str) -> list[str]:
    """取章型词表：无变体文种直接写列表（如调研报告），变体分键文种写 dict[变体→词表]。

    两种形态由各文种 ssot-config 自声明；dict 形态按解析后的变体键取值，
    列表形态对全部变体一体生效（无变体文种的词表不强制套变体分键结构）。
    """
    value = tmpl.get(key)
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return value.get(variant) or []
    return []


@register
def _rule_required_terms(ctx: _LintContext) -> list[Violation]:
    """术语必含词：按章型（+ 变体，如有分键）的必含词须出现于正文。"""
    tmpl = resolve_template(ctx.cfg, ctx.template)
    if not tmpl:
        return []
    required = _template_terms(tmpl, "required_terms", ctx.variant)
    out: list[Violation] = []
    for term in required:
        if term not in ctx.text:
            out.append(
                Violation(
                    rule="required_terms",
                    message=f"章型「{ctx.template}」({ctx.variant}) 须含术语「{term}」，但未出现。",
                )
            )
    return out


@register
def _rule_forbidden_terms(ctx: _LintContext) -> list[Violation]:
    """子风格措辞纯度：章型 + tier 的禁用词不得出现于正文。

    防跨层次渗入（如本科学制章混入高职「基本修业年限」、高职学制章混入
    本科「学士学位」）。词表来自 SSoT ``chapter_templates.<tmpl>.forbidden_terms``。
    """
    tmpl = resolve_template(ctx.cfg, ctx.template)
    if not tmpl:
        return []
    forbidden = _template_terms(tmpl, "forbidden_terms", ctx.variant)
    out: list[Violation] = []
    for term in forbidden:
        if term in ctx.text:
            out.append(
                Violation(
                    rule="forbidden_terms",
                    message=f"章型「{ctx.template}」({ctx.variant}) 禁用措辞「{term}」，属另一层次子风格，不得渗入。",
                )
            )
    return out


_SUBSECTION_HEADING = re.compile(r"^[#]{3,4}\s+(.+?)\s*$")
# 子项起手前缀：`（一）` / `（十二）` / `1.` / `1、` / `1)`。
_SUBSECTION_NUMERAL_PREFIX = re.compile(
    r"^(?:[（(][一二三四五六七八九十]+[）)]|[0-9]+[.、)])\s*"
)


def _subsection_headings(text: str) -> list[str]:
    """提取 ``###``/``####`` 子项标题，去掉编号前缀（如「（一）思政」→「思政」）。"""
    out: list[str] = []
    for line in text.splitlines():
        m = _SUBSECTION_HEADING.match(line)
        if not m:
            continue
        heading = _SUBSECTION_NUMERAL_PREFIX.sub("", m.group(1)).strip()
        if heading:
            out.append(heading)
    return out


@register
def _rule_forbidden_subsection(ctx: _LintContext) -> list[Violation]:
    """禁独立子项：章型 + tier 的禁用子项标题不得作 ``###``/``####`` 子项。

    防子风格结构串用（高职培养规格不得有独立「思政」子项——思政并入素质；
    本科培养规格须有思政独立子项，故仅对高职禁用）。精确匹配以避免误伤
    含该字的合并子项（如「思政素质」）。词表来自 SSoT
    ``chapter_templates.<tmpl>.forbidden_subsection_terms``。
    """
    tmpl = resolve_template(ctx.cfg, ctx.template)
    if not tmpl:
        return []
    forbidden = _template_terms(tmpl, "forbidden_subsection_terms", ctx.variant)
    if not forbidden:
        return []
    out: list[Violation] = []
    for heading in _subsection_headings(ctx.text):
        for term in forbidden:
            if heading == term:
                out.append(
                    Violation(
                        rule="forbidden_subsection",
                        message=(
                            f"章型「{ctx.template}」({ctx.variant}) 禁设独立子项「{term}」"
                            "（该子项应并入对应子风格结构）。"
                        ),
                    )
                )
    return out


# 建议条目起手：`（一）` / `1.` / `1)` / `建议一：`——对策建议节无 #### 子项时按此逐行切条。
_ACCOUNTABILITY_ITEM_LINE = re.compile(
    r"^\s*(?:[（(][一二三四五六七八九十]+[）)]|[0-9]+[.、)]|建议[一二三四五六七八九十]+[：:、])"
)


def _accountability_items(section_text: str) -> list[str]:
    """把对策建议节切成逐条建议：优先 ``####`` 子项块，退而按编号条目行，再退整节为一条。

    范式（规范源）：「建议一：…（教务处牵头，各二级学院落实，2026年9月前启动）」——
    单条建议通常一行/一块内自含责任与时限，故按块/按行判定即够；
    整节散文无条目结构时整节作一条判定，保证规则不因排版形态而失效。
    """
    blocks = [text for _title, text in _split_level_blocks(section_text, "#### ", "### ")]
    if blocks:
        return blocks
    lines = [line for line in section_text.splitlines() if _ACCOUNTABILITY_ITEM_LINE.match(line)]
    if lines:
        return lines
    return [section_text]


@register
def _rule_accountability(ctx: _LintContext) -> list[Violation]:
    """对策建议「明责任、定时限」：每条建议须同句/同条给出牵头责任与完成时限。

    配置挂在章型上（``chapter_templates.<tmpl>.accountability``，目前仅调研报告
    「结论与对策建议」章声明）：``section`` 定位章内目标节（``### `` 级标题含该词），
    ``responsibility_pattern`` 查牵头责任（牵头/责任部门等），``deadline_pattern``
    查完成时限（年月/期限）。目标节整体缺失（被裁）同样在此报——
    必含词是全文子串匹配，章标题本身含「对策建议」即满足，兜不住节级缺失。
    """
    tmpl = resolve_template(ctx.cfg, ctx.template)
    acc = (tmpl or {}).get("accountability")
    if not acc:
        return []
    section_name = acc.get("section")
    resp_pat = acc.get("responsibility_pattern")
    ddl_pat = acc.get("deadline_pattern")
    if not (section_name and resp_pat and ddl_pat):
        return []
    # 去编号前缀后与节标题精确匹配（「### （二）对策建议」→「对策建议」）：
    # 含词式匹配会被「持续改进方向」等其他含该词的节标题误定位。
    section_text: str | None = None
    for sec_title, sec_text in _split_level_blocks(ctx.text, "### ", "## "):
        if _SUBSECTION_NUMERAL_PREFIX.sub("", sec_title).strip() == section_name:
            section_text = sec_text
            break
    if section_text is None:
        return [
            Violation(
                rule="accountability",
                message=(
                    f"章型「{ctx.template}」未检出「{section_name}」节（### 级），"
                    "对策建议节不可裁剪，每条建议须明责任、定时限。"
                ),
            )
        ]
    out: list[Violation] = []
    for item in _accountability_items(section_text):
        missing = []
        if not re.search(resp_pat, item):
            missing.append("牵头责任")
        if not re.search(ddl_pat, item):
            missing.append("完成时限")
        if missing:
            snippet = next(
                (line.strip() for line in item.splitlines() if line.strip()), ""
            )[:40]
            out.append(
                Violation(
                    rule="accountability",
                    message=(
                        f"对策建议条目「{snippet}」缺{('与'.join(missing))}，"
                        "每条建议须明责任（牵头/落实部门）、定时限（完成年月或期限）。"
                    ),
                )
            )
    return out


@register
def _rule_avoid_title(ctx: _LintContext) -> list[Violation]:
    """禁用同义词标题：章标题不得使用 glossary 标注的 avoid 别名作大章权威词。"""
    if not ctx.template:
        return []
    out: list[Violation] = []
    for entry in ctx.cfg.get("glossary", []):
        if ctx.variant not in (entry.get("tiers") or []):
            continue
        canonical = entry.get("term", "")
        for avoid in entry.get("avoid", []) or []:
            if ctx.template == avoid or ctx.template.startswith(avoid):
                out.append(
                    Violation(
                        rule="avoid_title",
                        message=f"章标题「{ctx.template}」使用禁用同义词「{avoid}」，应作「{canonical}」。",
                    )
                )
    return out


@register
def _rule_political_theory(ctx: _LintContext) -> list[Violation]:
    """(A) 政治理论表述双向校验：逐字串 + 归位(belongs_to/required_in) + 子风格/领域标签。

    - ``required_in`` 章内须逐字出现（归位章内必含），缺则 ``political_theory_missing``。
    - ``belongs_to`` 外不得出现（归位章外禁出），出则 ``political_theory_out_of_place``。
      ``belongs_to`` 与 ``required_in`` 之间的章（在归位但不强制）不校验。
    - 领域专属串（``domain`` 非空）出现在非该领域文本 → ``political_theory_wrong_domain``。
    词表来自 SSoT ``ideology.political_theory``。
    """
    ideology = (ctx.cfg.get("ideology") or {}).get("political_theory") or []
    out: list[Violation] = []
    chap = resolve_ideology_chapter(ctx.cfg, ctx.template)
    for entry in ideology:
        verbatim = entry.get("verbatim", "")
        belongs_to = entry.get("belongs_to") or []
        required_in = entry.get("required_in") or []
        entry_domain = entry.get("domain")
        present = verbatim in ctx.text
        # 领域专属串：出现在非该领域文本即违规（不区分子风格/tier）。
        if entry_domain is not None and entry_domain != ctx.domain:
            if present:
                out.append(
                    Violation(
                        rule="political_theory_wrong_domain",
                        message=(
                            f"领域专属政治语「{verbatim}」不得出现在非「{entry_domain}」领域文档。"
                        ),
                    )
                )
            continue
        # 通用串或领域匹配：归位校验按 tier 过滤。
        if ctx.variant not in (entry.get("tiers") or []):
            continue
        if chap in required_in and not present:
            out.append(
                Violation(
                    rule="political_theory_missing",
                    message=(
                        f"归位章「{ctx.template}」({ctx.variant}) 须逐字含「{verbatim}」，"
                        "但未出现（不得改写或引错）。"
                    ),
                )
            )
        elif chap not in belongs_to and present:
            out.append(
                Violation(
                    rule="political_theory_out_of_place",
                    message=(
                        f"政治理论表述「{verbatim}」只应出现在归位章"
                        f"（{belongs_to}），不得注入「{ctx.template}」章。"
                    ),
                )
            )
    return out


@register
def _rule_affective(ctx: _LintContext) -> list[Violation]:
    """(B) 价值导向/情感语在位校验（略松）。

    培养目标/素质章（``affective.required_in``）须含至少一条情感语正则
    （``affective.patterns``，如「工匠精神」「报国」「爱国」「强国志」）。
    用正则 ``search`` 实现略松——容忍「科技报国意识」「爱国情怀」等小幅措辞变化。
    词表来自 SSoT ``ideology.affective``。
    """
    affective = (ctx.cfg.get("ideology") or {}).get("affective") or {}
    if not affective:
        return []
    if ctx.variant not in (affective.get("tiers") or []):
        return []
    chap = resolve_ideology_chapter(ctx.cfg, ctx.template)
    if chap not in (affective.get("required_in") or []):
        return []
    patterns = affective.get("patterns") or []
    if not any(re.search(p, ctx.text) for p in patterns):
        return [
            Violation(
                rule="affective_missing",
                message=(
                    f"归位章「{ctx.template}」({ctx.variant}) 须含价值导向/情感语"
                    f"（{patterns}），但均未出现。"
                ),
            )
        ]
    return []


@register
def _rule_political_theory_partial(ctx: _LintContext) -> list[Violation]:
    """(A) 逐字精确复现：检测「引用了但未逐字」的部分提及，与 ``required_in`` 解耦。

    语料实证表明某些逐字串（如「习近平新时代中国特色社会主义思想」）并非每篇归位章
    都逐字出现（课程设置章多以「概论」简称），故不能一律 ``required_in`` 强制必含——
    全不提的章不报。但「逐字精确复现」是命脉：若正文引用了该理论（命中
    ``partial_trigger``，如「习近平」）却未逐字精确出现全串（常见删字/断词改写），
    仍须检出。规则：``partial_trigger`` 在正文、但 ``verbatim`` 不在（归一化 CJK 断词
    后比对）→ ``political_theory_partial``。``lint`` 入口已对 text 做 CJK 断词归一，
    故 PDF 抽取的「新 时代」断词不致假性触发。
    """
    out: list[Violation] = []
    for entry in (ctx.cfg.get("ideology") or {}).get("political_theory") or []:
        trigger = entry.get("partial_trigger")
        if not trigger:
            continue
        if ctx.variant not in (entry.get("tiers") or []):
            continue
        ed = entry.get("domain")
        if ed is not None and ed != ctx.domain:
            continue
        verbatim = entry.get("verbatim", "")
        if verbatim in ctx.text:
            continue
        if trigger in ctx.text:
            out.append(
                Violation(
                    rule="political_theory_partial",
                    message=(
                        f"正文引用了「{trigger}」但未逐字精确出现「{verbatim}」，"
                        "政治理论表述须逐字精确复现，不得改写或引错。"
                    ),
                )
            )
    return out


@register
def _rule_reference_present(ctx: _LintContext) -> list[Violation]:
    """引用事实在位：``references`` 里每个 Fact.value 须出现于正文。

    防写作漏写调用方传入的事实依据（行业代码/课程/学分/证书）。``value`` 作
    子串匹配（``in``）；数值类事实（``_NUMERIC_FACT_TYPES``）两侧先做
    ``normalize_numeric_text`` 归一（全角转半角、去空格）再比对，
    修排版差异漏判。``references`` 为空/None 则不校验（未补全事实依据时不误伤）。
    事实来源由调用方传入，非 SSoT。
    """
    if not ctx.references:
        return []
    numeric_text: str | None = None
    out: list[Violation] = []
    for fact in ctx.references:
        if fact.type in _NUMERIC_FACT_TYPES:
            if numeric_text is None:
                numeric_text = normalize_numeric_text(ctx.text)
            present = normalize_numeric_text(fact.value) in numeric_text
        else:
            present = fact.value in ctx.text
        if not present:
            out.append(
                Violation(
                    rule="reference_missing",
                    message=(
                        f"引用事实 [{fact.type}]「{fact.value}」未出现于正文，"
                        "须忠实融入调用方传入的事实依据。"
                    ),
                )
            )
    return out


@register
def _rule_unknown_material_marker(ctx: _LintContext) -> list[Violation]:
    """无杜撰素材：正文每个 ``[素材id]`` 角标的 id 须存在于素材池内。

    写作子智能体只许引用素材池内条目，正文角标出现池外 id 即为杜撰来源。
    角标由 ``MARKER_PATTERN`` 抽取；同一 id 重复出现只报一次。
    ``materials`` 为空/None 则不校验（不误伤无素材语料）。
    """
    if not ctx.materials:
        return []
    pool_ids = {material["id"] for material in ctx.materials}
    out: list[Violation] = []
    seen: set[str] = set()
    for marker in MARKER_PATTERN.findall(ctx.text):
        if marker in seen:
            continue
        seen.add(marker)
        if marker not in pool_ids:
            out.append(
                Violation(
                    rule="unknown_material_marker",
                    message=(
                        f"正文角标 [{marker}] 不在素材池内（池：{sorted(pool_ids)}），"
                        "禁止杜撰/篡改素材来源，仅可引用池内素材 id。"
                    ),
                )
            )
    return out


@register
def _rule_duplicate_material_id(ctx: _LintContext) -> list[Violation]:
    """素材池内无重复 id：``materials`` 内每条 id 唯一。

    素材身份须稳定（id 由上游分配、跨章不变），同一池内出现重复 id 即身份冲突。
    仅检测池内重复，与正文角标无关（正文复用同 id 是允许的）。
    ``materials`` 为空/None 则不校验（不误伤无素材语料）。
    """
    if not ctx.materials:
        return []
    seen: set[str] = set()
    out: list[Violation] = []
    for material in ctx.materials:
        if material["id"] in seen:
            out.append(
                Violation(
                    rule="duplicate_material_id",
                    message=(
                        f"素材池内出现重复 id「{material['id']}」，"
                        "素材身份须唯一稳定（上游分配，不得重复登记）。"
                    ),
                )
            )
            continue
        seen.add(material["id"])
    return out


@register
def _rule_dangling_hypothesis_id(ctx: _LintContext) -> list[Violation]:
    """hypothesis_id 合法：每条素材的 ``hypothesis_id`` 须存在于本章假说列表。

    素材须逐条回链假说，指向不存在的假说即为悬空。仅校验池内每条素材的绑定，
    与正文角标无关。``materials`` 为空/None 或 ``hypotheses`` 为空/None 则不校验
    （无池/无假说列表可对照）。
    """
    if not ctx.materials or not ctx.hypotheses:
        return []
    hyp_ids = {hypothesis["id"] for hypothesis in ctx.hypotheses}
    out: list[Violation] = []
    seen: set[str] = set()
    for material in ctx.materials:
        if material["hypothesis_id"] in seen:
            continue
        seen.add(material["hypothesis_id"])
        if material["hypothesis_id"] not in hyp_ids:
            out.append(
                Violation(
                    rule="dangling_hypothesis_id",
                    message=(
                        f"素材「{material['id']}」的 hypothesis_id「{material['hypothesis_id']}」"
                        f"不在假说列表（假说：{sorted(hyp_ids)}），素材须回链既有假说。"
                    ),
                )
            )
    return out


# 照抄型「派生未标」守卫：提取素材 excerpt 特征片段时剔除的标点/空白。
# 覆盖 CJK 标点 + ASCII 标点（含 []() 等），仅服务本规则的去标点比对，
# 不影响 lint 主归一化——``text`` 仍由 ``lint()`` 预 ``normalize_cjk_ws``。
_DERIVED_PUNCT = frozenset("。，、；：！？·…—（）「」『』《》〈〉“”‘’") | frozenset(string.punctuation)

# 特征片段最短长度：短于此的 excerpt 视为无可用特征（避免「人才」等极短串误报）。
_MIN_DERIVED_SIG_LEN = 6


def _excerpt_signature(excerpt: str) -> str | None:
    """从素材 ``excerpt`` 提取稳定特征片段：CJK 空白归一化 + 去标点/空白后取整段。

    返回去标点/空白后的连续串；不足 ``_MIN_DERIVED_SIG_LEN`` 视为无可用特征（``None``），
    规则对该素材不触发——极短片段不足以构成照抄判据。
    继承 ``lint()`` 已做的 CJK 空白归一化（``text`` 端已归一，此处对 excerpt 同口径处理）。
    """
    norm = normalize_cjk_ws(excerpt)
    sig = "".join(ch for ch in norm if ch not in _DERIVED_PUNCT and not ch.isspace())
    return sig if len(sig) >= _MIN_DERIVED_SIG_LEN else None


@register
def _rule_unmarked_derived_content(ctx: _LintContext) -> list[Violation]:
    """无未标派生（照抄型）：素材 ``excerpt`` 特征片段出现于正文却无对应 ``[素材id]`` 角标则报违规。

    与 ``_rule_reference_present`` 同族（子串式），但反向：
    该素材的摘录特征片段 ``in`` 正文、且其 id 未作为 ``[素材id]`` 角标出现于正文 →
    视为「照抄原文却漏标」。特征片段经 ``_excerpt_signature`` 去标点/空白后取整段比对，
    正文同口径去标点/空白（``text`` 已由 ``lint()`` 预 ``normalize_cjk_ws``），
    容忍标点/断词轻微漂移。仅抓近乎照抄；改写型确定性抓不到，留待后续自审环节。
    ``excerpt`` 为空或过短（无可用特征）→ 该条不触发；
    ``materials`` 为空/None 则不校验（不误伤无素材语料）。
    """
    if not ctx.materials:
        return []
    marked_ids = set(MARKER_PATTERN.findall(ctx.text))
    # 正文去标点/空白后的归一串，供特征片段子串比对（与 _excerpt_signature 同口径）。
    text_norm = "".join(
        ch for ch in ctx.text if ch not in _DERIVED_PUNCT and not ch.isspace()
    )
    out: list[Violation] = []
    for material in ctx.materials:
        if not material["excerpt"]:
            continue
        sig = _excerpt_signature(material["excerpt"])
        if sig is None or material["id"] in marked_ids:
            continue
        if sig in text_norm:
            out.append(
                Violation(
                    rule="unmarked_derived_content",
                    message=(
                        f"素材「{material['id']}」的 excerpt 特征片段「{sig}」"
                        "出现于正文却无对应 [素材id] 角标，照抄原文处须挂角标。"
                    ),
                )
            )
    return out


def _extract_tokens(text: str, patterns: list[str], group: int = 1) -> set[str]:
    """对 ``patterns`` 逐条 ``finditer``，取指定捕获组（默认 1；0 取整段匹配）。"""
    tokens: set[str] = set()
    for pat in patterns or []:
        for m in re.finditer(pat, text):
            try:
                tok = m.group(group)
            except IndexError:  # 该 pattern 无此捕获组 → 降级取整段匹配
                tok = m.group(0)
            if tok:
                tokens.add(tok)
    return tokens


def extract_facts(text: str, cfg: dict[str, Any]) -> list[Fact]:
    """从正文抽结构化事实 token（行业代码 / 学分 / 证书名），按 SSoT ``fabrication`` 节。

    与 ``_rule_fabrication`` 共享同一抽取以零漂移：正文抽 token 与 ``references``
    比对查臆造；后续也可供反向提取事实依据复用。
    证书名在此一并应用 ``allowlist`` + ``prose_markers`` 过滤，使各消费路径口径一致。
    """
    fab = cfg.get("fabrication") or {}
    facts: list[Fact] = []

    spec = fab.get("industry_code") or {}
    pat = spec.get("pattern")
    if pat:
        for tok in _extract_tokens(text, [pat], int(spec.get("group", 1))):
            facts.append(Fact(type="industry_code", value=tok))

    spec = fab.get("credit") or {}
    for tok in _extract_tokens(text, spec.get("patterns") or [], 1):
        facts.append(Fact(type="credit", value=tok))

    spec = fab.get("certificate") or {}
    pat = spec.get("pattern")
    if pat:
        allowlist = set(spec.get("allowlist") or [])
        prose_markers = spec.get("prose_markers") or []
        for tok in _extract_tokens(text, [pat], 0):
            tok_s = tok.strip()
            if tok_s in allowlist:
                continue
            if any(marker in tok_s for marker in prose_markers):
                continue
            facts.append(Fact(type="certificate", value=tok_s))

    return facts


# 量化断言的同句判定所用句末标点（角标可紧随句末标点之后，仍算同句）。
_SENTENCE_TERMINATORS = "。！？；"


def _sentence_with_trailing_markers(line: str, start: int, end: int) -> str:
    """取 ``line[start:end]`` 所在句，句末标点后紧随的 ``[素材id]`` 角标一并计入。

    句边界按 ``_SENTENCE_TERMINATORS`` 划定；语料中角标既见于句内
    （``…提升30%[m-h-1]。``）也见于紧随句末标点（``…提升30%。[m-h-1]``），
    两种写法均属该句的溯源标注，故向后扩展吞并紧随的角标序列。
    """
    begin = max(line.rfind(ch, 0, start) for ch in _SENTENCE_TERMINATORS) + 1
    stops = [i for i in (line.find(ch, end) for ch in _SENTENCE_TERMINATORS) if i != -1]
    stop = min(stops) + 1 if stops else len(line)
    trailing = re.match(r"(?:\s*\[[A-Za-z0-9_\-]+\])+", line[stop:])
    if trailing:
        stop += trailing.end()
    return line[begin:stop]


def _quantitative_violations(ctx: _LintContext, fab: dict[str, Any]) -> list[Violation]:
    """量化断言查臆造（``fabrication.quantitative`` 子类型）。

    正文散文出现「提升/降低/缩短/增长/减少 + 数值单位」的量化断言时，
    须同句挂 ``[素材id]`` 角标、或断言数值（``value_pattern`` 抽取）能在任一
    reference value（数值归一化后）中找到依据，二者皆无则违规。
    ``references_exempt: false``（严格文种经标量覆盖设定，如调研报告）时收紧为
    必须同句角标，references 数值有据不再豁免——溯源在正文可见。
    表行（``|`` 起手）不抽取——表内数字由表承载，另有表规则管。
    """
    spec = fab.get("quantitative") or {}
    pat = spec.get("pattern")
    if not pat:
        return []
    value_pat = spec.get("value_pattern")
    references_exempt = spec.get("references_exempt", True)
    backed_nums: set[str] = set()
    if value_pat and references_exempt:
        for fact in ctx.references or []:
            backed_nums.update(re.findall(value_pat, normalize_numeric_text(fact.value)))
    out: list[Violation] = []
    for line in ctx.text.splitlines():
        if line.lstrip().startswith("|"):
            continue
        for m in re.finditer(pat, line):
            sentence = _sentence_with_trailing_markers(line, m.start(), m.end())
            if MARKER_PATTERN.search(sentence):
                continue
            num = re.search(value_pat, m.group(0)) if value_pat and references_exempt else None
            if num and num.group(0) in backed_nums:
                continue
            message = (
                f"量化断言「{m.group(0)}」无同句素材角标、数值亦无 references 依据，"
                "疑为臆造，须挂角标或改用有据数值。"
                if references_exempt
                else f"数据断言「{m.group(0)}」无同句素材角标，"
                "本文种要求溯源在正文可见，须同句挂 [素材id] 角标。"
            )
            out.append(Violation(rule="fabricated_quantitative", message=message))
    return out


@register
def _rule_fabrication(ctx: _LintContext) -> list[Violation]:
    """查臆造：正则从正文抽候选 token，须能在 references 找到依据。

    按类型（industry_code/credit/certificate）分别校验；仅当 references 含该类型
    ≥1 条时才校验该类型——调用方未提供某类型依据即视为该类型不在本章事实范围内，
    不误伤未补全 references 的章。token 抽取委托 ``extract_facts``，
    模式来自 SSoT ``fabrication``，避免校验器与指南漂移。
    quantitative 子类型另有素材角标这一备用依据通道，故只要调用方提供了
    references 或 materials 之一即校验（两者皆无仍整组跳过，不误伤裸语料）。
    """
    if not ctx.references and not ctx.materials:
        return []
    fab = ctx.cfg.get("fabrication") or {}
    out: list[Violation] = _quantitative_violations(ctx, fab)
    if not ctx.references:
        return out
    by_type: dict[str, list[Fact]] = {}
    for f in ctx.references:
        by_type.setdefault(f.type, []).append(f)
    text_facts = extract_facts(ctx.text, ctx.cfg)

    ic_refs = by_type.get("industry_code")
    if ic_refs:
        backed = {f.value for f in ic_refs}
        for tf in text_facts:
            if tf.type == "industry_code" and tf.value not in backed:
                out.append(
                    Violation(
                        rule="fabricated_industry_code",
                        message=f"正文出现行业代码「{tf.value}」无 references 依据，疑为臆造。",
                    )
                )

    cr_refs = by_type.get("credit")
    if cr_refs:
        value_pat = (fab.get("credit") or {}).get("value_pattern")
        backed_nums: set[str] = set()
        for f in cr_refs:
            if value_pat:
                m = re.search(value_pat, f.value)
                if m:
                    backed_nums.add(m.group(0))
            else:
                backed_nums.add(f.value)
        for tf in text_facts:
            if tf.type == "credit" and tf.value not in backed_nums:
                out.append(
                    Violation(
                        rule="fabricated_credit",
                        message=f"正文出现学分「{tf.value}」无 references 依据，疑为臆造。",
                    )
                )

    ce_refs = by_type.get("certificate")
    if ce_refs:
        cert_values = [f.value for f in ce_refs]
        for tf in text_facts:
            if tf.type == "certificate":
                cert_backed = any(
                    tf.value == r or tf.value in r or r in tf.value for r in cert_values
                )
                if not cert_backed:
                    out.append(
                        Violation(
                            rule="fabricated_certificate",
                            message=f"正文出现证书名「{tf.value}」无 references 依据，疑为臆造。",
                        )
                    )
    return out


def count_prose_words(text: str) -> int:
    """统计正文散文字数（纯函数）：排除表格、素材角标、公式、代码块、参考文献、附录。

    统计口径：
    - 排除整个 markdown 表格（含列头、分隔行、数据行）。
    - 排除素材角标 ``[素材id]``（复用 MARKER_PATTERN）。
    - 排除行内与块级公式（``$...$`` / ``$$...$$``）。
    - 排除围栏代码块（````...````）。
    - 排除参考文献与附录（``## 参考文献`` / ``## 附录`` 之后的全部内容）。
    - markdown 标题行不做单独排除判断（标题行内容计入字数）。

    仅统计汉字（按字计）+ 半角字母数字词（连续字母数字为一词），
    标点与空白一律不计。返回总字数。
    """
    # 1. 排除参考文献/附录：从首次出现 ``## 参考文献`` 或 ``## 附录`` 处截断。
    for cutoff in ["\n## 参考文献", "\n## 附录"]:
        pos = text.find(cutoff)
        if pos != -1:
            text = text[:pos]

    # 2. 排除围栏代码块：````...````（多行）。
    text = re.sub(r"```[\s\S]*?```", "", text)

    # 3. 排除行内与块级公式：``$...$`` / ``$$...$$``。
    text = re.sub(r"\$\$[\s\S]*?\$\$", "", text)
    text = re.sub(r"\$[^\$\n]+?\$", "", text)

    # 4. 排除素材角标：``[素材id]``（复用 MARKER_PATTERN）。
    text = MARKER_PATTERN.sub("", text)

    # 5. 排除整个 markdown 表格（含列头、分隔行、数据行）：逐行扫描，遇分隔行
    # 开始表区、向前回溯删除列头、向后删除数据行直至非表行。
    lines = text.splitlines(keepends=True)
    table_ranges: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        if _MD_TABLE_SEP.match(lines[i]):
            # 分隔行命中：向前回溯列头行（起始 | 的行，允许多行列头）。
            start = i
            while start > 0 and lines[start - 1].strip().startswith("|"):
                start -= 1
            # 向后吞数据行（起始 | 的行，直至非表行）。
            end = i + 1
            while end < len(lines) and lines[end].strip().startswith("|"):
                end += 1
            table_ranges.append((start, end))
            i = end
        else:
            i += 1
    # 倒序删除表区（避免索引偏移）。
    for start, end in reversed(table_ranges):
        del lines[start:end]
    text = "".join(lines)

    # 6. 统计：汉字按字计 + 半角字母数字词计数（连续的字母数字为一词），
    # 标点（中英文）与空白一律不计。
    cjk_chars = len(re.findall(r"[一-鿿㐀-䶿]", text))
    alnum_words = len(re.findall(r"[a-zA-Z0-9]+", text))
    return cjk_chars + alnum_words


def _split_level_blocks(text: str, heading_prefix: str, stop_prefix: str) -> list[tuple[str, str]]:
    """按 ``heading_prefix`` 级标题切块：返回 (标题, 块正文含标题行) 列表。

    块从该级标题行起，到下一个同级标题、或上一级标题（``stop_prefix``）、
    或文末止。``heading_prefix`` 形如 ``### ``，``stop_prefix`` 形如 ``## ``
    （节块遇下一章标题即止；小节块遇节标题即止）。
    """
    lines = text.splitlines()
    blocks: list[tuple[str, str]] = []
    current_title: str | None = None
    current_lines: list[str] = []
    for line in lines:
        is_level = line.startswith(heading_prefix)
        is_stop = not is_level and line.startswith(stop_prefix)
        if is_level or is_stop:
            if current_title is not None:
                blocks.append((current_title, "\n".join(current_lines)))
                current_title = None
                current_lines = []
            if is_level:
                current_title = line[len(heading_prefix):].strip()
                current_lines = [line]
            continue
        if current_title is not None:
            current_lines.append(line)
    if current_title is not None:
        blocks.append((current_title, "\n".join(current_lines)))
    return blocks


def _shrunk_range(
    cfg_min: float,
    cfg_max: float,
    parent_min: float,
    parent_max: float,
    siblings: int,
    ratio: float,
) -> tuple[float, float]:
    """节/小节区间按同级数量动态收缩：上限取「配置上限」与「父级上限÷同级数量」较小值，
    下限取「配置下限×折减系数」与「父级下限÷同级数量」较大值；收缩后保证下限不高于上限。
    父级区间取直接上级（节←章、小节←节），否则区间会退化（见 ADR-0003）。
    """
    upper = min(cfg_max, parent_max / siblings)
    lower = max(cfg_min * ratio, parent_min / siblings)
    return min(lower, upper), upper


def check_word_count(
    text: str, cfg: dict[str, Any], *, template_title: str | None = None
) -> list[Violation]:
    """字数管控核心校验（纯函数）：三级区间 + 动态收缩 + 节级同级差异 + 表章豁免。

    经 lint 注册规则消费（修一次后的复检直接复用全量 lint，同一口径零漂移）。
    ``cfg`` 无 ``word_count`` 节、或正文无 ``## `` 章标题（非标准章结构）时不校验。
    表章（``table_required`` 章型）豁免各级散文下限与节级同级差异比对，仅保各级上限。
    ``template_title`` 显式指定章型标题（State 携带的章型，如观点标题的维度章）；
    未指定则回落用正文标题反推的章型解析豁免。
    """
    wc = cfg.get("word_count")
    if not wc:
        return []
    title = detect_chapter_template(text, cfg)
    if title is None:
        return []
    tmpl = resolve_template(cfg, template_title if template_title is not None else title)
    table_exempt = bool(tmpl and tmpl.get("table_required"))
    ch_min = float(wc["chapter"]["min"])
    ch_max = float(wc["chapter"]["max"])
    ratio = float(wc.get("min_shrink_ratio", 0.7))
    balance_ratio = float(wc.get("section_balance_max_ratio", 2))
    out: list[Violation] = []

    # 章级：最终硬标准。表章豁免下限、仅保上限。
    chapter_count = count_prose_words(text)
    if chapter_count > ch_max:
        out.append(
            Violation(
                rule="word_count",
                message=(
                    f"章「{title}」散文 {chapter_count} 字超出上限 {ch_max:.0f} 字，"
                    "须压缩正文、去除注水内容。"
                ),
            )
        )
    elif not table_exempt and chapter_count < ch_min:
        out.append(
            Violation(
                rule="word_count",
                message=(
                    f"章「{title}」散文 {chapter_count} 字不足下限 {ch_min:.0f} 字，"
                    "须充分论证每个论点、结合假说展开、给出依据与技术路径。"
                ),
            )
        )

    # 节级：按同级数量动态收缩后校验。
    sections = _split_level_blocks(text, "### ", "## ")
    section_counts: list[tuple[str, int]] = [
        (sec_title, count_prose_words(sec_text)) for sec_title, sec_text in sections
    ]
    if sections:
        sec_lower, sec_upper = _shrunk_range(
            float(wc["section"]["min"]), float(wc["section"]["max"]),
            ch_min, ch_max, len(sections), ratio,
        )
        for sec_title, sec_count in section_counts:
            if sec_count > sec_upper:
                out.append(
                    Violation(
                        rule="word_count",
                        message=(
                            f"节「{sec_title}」散文 {sec_count} 字超出上限 {sec_upper:.0f} 字"
                            f"（{len(sections)} 节动态收缩后区间），须压缩该节。"
                        ),
                    )
                )
            elif not table_exempt and sec_count < sec_lower:
                out.append(
                    Violation(
                        rule="word_count",
                        message=(
                            f"节「{sec_title}」散文 {sec_count} 字不足下限 {sec_lower:.0f} 字"
                            f"（{len(sections)} 节动态收缩后区间），须充分展开该节论证。"
                        ),
                    )
                )

    # 节级同级差异：最长 ≤ 最短 × 倍数（只查章内节级；表章豁免；不足两节不比）。
    if not table_exempt and len(section_counts) >= 2:
        shortest_title, shortest = min(section_counts, key=lambda pair: pair[1])
        longest_title, longest = max(section_counts, key=lambda pair: pair[1])
        if shortest > 0 and longest > shortest * balance_ratio:
            out.append(
                Violation(
                    rule="word_count",
                    message=(
                        f"章内节级体量失衡：最长节「{longest_title}」{longest} 字超过"
                        f"最短节「{shortest_title}」{shortest} 字的 {balance_ratio:.0f} 倍，"
                        "须均衡各节展开程度。"
                    ),
                )
            )

    # 小节级：在各节内部按同级数量动态收缩后校验。
    # 父级取节区间而非章区间：小节的体量约束来自其所在节；若误用章区间，
    # 「章下限÷小节数」在常见小节数（2～4）下会把下限顶到配置上限，
    # 区间退化为点区间，正常体量的小节被误判、修一次循环永不收敛
    # （issue #19 真实 E2E 复跑发现）。
    for sec_title, sec_text in sections:
        subsections = _split_level_blocks(sec_text, "#### ", "### ")
        if not subsections:
            continue
        sub_lower, sub_upper = _shrunk_range(
            float(wc["subsection"]["min"]), float(wc["subsection"]["max"]),
            float(wc["section"]["min"]), float(wc["section"]["max"]),
            len(subsections), ratio,
        )
        for sub_title, sub_text in subsections:
            sub_count = count_prose_words(sub_text)
            if sub_count > sub_upper:
                out.append(
                    Violation(
                        rule="word_count",
                        message=(
                            f"小节「{sub_title}」散文 {sub_count} 字超出上限 {sub_upper:.0f} 字"
                            f"（{len(subsections)} 小节动态收缩后区间），须压缩该小节。"
                        ),
                    )
                )
            elif not table_exempt and sub_count < sub_lower:
                out.append(
                    Violation(
                        rule="word_count",
                        message=(
                            f"小节「{sub_title}」散文 {sub_count} 字不足下限 {sub_lower:.0f} 字"
                            f"（{len(subsections)} 小节动态收缩后区间），须充分展开该小节论证。"
                        ),
                    )
                )
    return out


@register
def _rule_word_count(ctx: _LintContext) -> list[Violation]:
    """字数管控：三级区间（章/节/小节）+ 动态收缩 + 节级同级差异 + 表章豁免。

    校验逻辑收敛于 ``check_word_count``（与「修一次后复检」双消费零漂移）；
    区间配置来自 SSoT ``word_count`` 节，未配置或非标准章结构不校验。
    章型取 ``ctx.template``（State 携带优先），使观点标题的维度章也享表章豁免。
    """
    return check_word_count(ctx.text, ctx.cfg, template_title=ctx.template)


def word_count_prompt_block(
    title: str,
    doc_type: str,
    *,
    style_guides_dir: str | Path | None = None,
    chapter_type: str | None = None,
) -> str:
    """按章标题生成写作提示词的目标字数区间块；SSoT 无 ``word_count`` 配置时返回空串。

    区间与章型模板取自该文种的两层合并配置（``load_config``）——
    表章（``table_required`` 章型）提示取中下限且不得表外堆砌；
    叙述章型提示取中上限。区间数值全部取自 SSoT，提示词与校验器零漂移。
    ``chapter_type`` 是 State 携带的章型（ADR-0005），传入时优先据其解析章型模板
    （观点标题的维度章无法靠标题匹配）；未传回落标题解析。
    """
    cfg = load_config(doc_type, style_guides_dir=style_guides_dir)
    wc = cfg.get("word_count")
    if not wc:
        return ""
    normalized_title = _CHINESE_NUMERAL_PREFIX.sub("", title).strip()
    tmpl = resolve_template(cfg, chapter_type if chapter_type is not None else normalized_title)
    table_chapter = bool(tmpl and tmpl.get("table_required"))
    ch, sec, sub = wc["chapter"], wc["section"], wc["subsection"]
    lines = [
        "本章目标字数（散文统计口径：不含表格、素材角标、公式、代码块）：",
        f"- 章总量 {ch['min']}～{ch['max']} 字；节（###）每节 {sec['min']}～{sec['max']} 字；"
        f"小节（####）每小节 {sub['min']}～{sub['max']} 字。",
    ]
    if table_chapter:
        lines.append(
            "- 本章为表型章：信息由表承载，散文只做引言与解读，散文体量取区间中下限即可，"
            "不得在表外堆砌叙述性段落凑字数。"
        )
    else:
        lines.append(
            "- 本章为叙述章型：正文体量宜取区间中上限，同章各节展开程度须均衡"
            "（最长节不超过最短节的 2 倍）。"
        )
    lines.append(
        "- 字数区间是机检硬标准：任何一级（章/节/小节）超出区间即判违规并触发整章返工，"
        "成文前须逐级自查各章节字数落在区间内。"
    )
    return "\n".join(lines)


def lint(
    text: str,
    doc_type: str,
    doc_variant: str | None = None,
    *,
    style_guides_dir: str | Path | None = None,
    chapter_type: str | None = None,
    domain: str | None = None,
    references: list[Fact] | None = None,
    materials: list[MaterialPayload] | None = None,
    hypotheses: list[HypothesisPayload] | None = None,
) -> list[Violation]:
    """对单章正文跑全部已注册规则，返回违规列表（纯函数）。

    规则集按 ``doc_type`` 两层合并加载（通用层 + 文种层，ADR-0005）；
    ``doc_variant`` 是文种内变体（人培方案的 本科/高职），经 ``tier_from_variant``
    解析为变体分键规则的取值键——变体概念仅存活于人培方案内部，其他文种的
    合并配置不含变体分键规则，该参数对其天然惰性。
    ``style_guides_dir`` 缺省用随包 ``style_guides/`` 目录（不依赖工作目录）；
    显式传目录便于测试与替换指南。
    ``chapter_type`` 是 State 大纲随章携带的章型（ADR-0005，骨架事实）：传入时
    章型规则直接据其匹配 ``chapter_templates``，不从位置或标题反推——
    维度章的自由观点标题唯有经此参数才能命中章型规则；
    未传（自由结构/旧链路）回落从正文首个 ``## `` 标题反推。
    ``domain`` 标注被校验文本所属领域（如「金融」；通用为 None），
    供意识形态 (A) 的领域专属政治语双向校验（规则级条件标签，非加载参数）。
    ``references`` 为本章调用方传入的事实依据（``Fact`` 列表），供「引用事实在位 +
    查臆造」校验；为空/None 则跳过该两条规则（不误伤未补全事实依据的章）。
    ``materials`` 为本章素材池（``MaterialPayload`` 列表），``hypotheses`` 为本章假说列表，
    供素材相关结构校验（无杜撰角标 / 池内 id 唯一 / hypothesis_id 合法 / 照抄型派生未标）；
    为空/None 则跳过素材相关规则（不误伤无素材语料）。
    """
    cfg = load_config(doc_type, style_guides_dir=style_guides_dir)
    normalized = normalize_cjk_ws(text)
    ctx = _LintContext(
        text=normalized,
        cfg=cfg,
        variant=tier_from_variant(doc_variant),
        template=(
            chapter_type
            if chapter_type is not None
            else detect_chapter_template(normalized, cfg)
        ),
        domain=domain,
        references=references,
        materials=materials,
        hypotheses=hypotheses,
    )
    violations: list[Violation] = []
    for rule in _RULES:
        violations.extend(rule(ctx))
    return violations
