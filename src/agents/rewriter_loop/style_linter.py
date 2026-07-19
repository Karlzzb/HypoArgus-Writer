"""style_linter：单章正文的规则校验器（纯函数，不依赖主图与 LLM）。

入口签名：``lint(text, tier, *, style_guide_path=None, ...) -> list[Violation]``。
词表/禁用词/boilerplate 从随包 ``style_guide.md`` 的 SSoT YAML 块解析加载（单一事实源）；
机械规则（口语化黑名单、编号）的词表同样来自该 YAML 块，避免指南与校验器漂移。
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

# 随包携带的风格指南默认路径：不依赖工作目录，显式传参可覆盖（便于测试）。
DEFAULT_STYLE_GUIDE_PATH = Path(__file__).parent / "style_guide.md"

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


@dataclass(frozen=True)
class _LintContext:
    """一次 lint 调用中传给各规则的共享输入（正文已做 CJK 断词归一）。"""

    text: str
    cfg: dict[str, Any]
    tier: str
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


def load_config(style_guide_path: str | Path | None = None) -> dict[str, Any]:
    """从 style_guide.md 解析 SSoT YAML 块（``<!-- ssot-config-begin ... -end -->`` 之间）。"""
    path = Path(style_guide_path) if style_guide_path is not None else DEFAULT_STYLE_GUIDE_PATH
    text = path.read_text(encoding="utf-8")
    m = re.search(r"ssot-config-begin\s*\n(.*?)\n\s*ssot-config-end", text, re.S)
    if not m:
        raise ValueError(f"style_guide 未找到 ssot-config 块：{path}")
    config = yaml.safe_load(m.group(1))
    if not isinstance(config, dict):
        raise ValueError(f"style_guide 的 ssot-config 块须为映射：{path}")
    return config


def load_prose(style_guide_path: str | Path | None = None) -> str:
    """读取 style_guide.md 散文部分，供注入写作提示词用。

    YAML 块（``<!-- ssot-config-begin ... -end -->``）是校验器的机器可读编码，
    散文已用人话描述同样规则，故只注入散文以节省上下文；
    YAML 仍由 ``load_config`` 独立解析，同一文件双消费、零漂移。
    """
    path = Path(style_guide_path) if style_guide_path is not None else DEFAULT_STYLE_GUIDE_PATH
    text = path.read_text(encoding="utf-8")
    idx = text.find("<!-- ssot-config-begin")
    return text[:idx] if idx != -1 else text


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
    """禁讲客体/口语化表达（黑名单来自 SSoT YAML）。"""
    out: list[Violation] = []
    for term in ctx.cfg.get("oral_blacklist", []):
        if term in ctx.text:
            out.append(
                Violation(
                    rule="oral_blacklist",
                    message=f"出现讲客体/口语化表达「{term}」，公文须用第三人称/无主语祈使。",
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


@register
def _rule_required_terms(ctx: _LintContext) -> list[Violation]:
    """术语必含词：按章型 + tier 的必含词须出现于正文。"""
    tmpl = resolve_template(ctx.cfg, ctx.template)
    if not tmpl:
        return []
    required = (tmpl.get("required_terms") or {}).get(ctx.tier, [])
    out: list[Violation] = []
    for term in required:
        if term not in ctx.text:
            out.append(
                Violation(
                    rule="required_terms",
                    message=f"章型「{ctx.template}」({ctx.tier}) 须含术语「{term}」，但未出现。",
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
    forbidden = (tmpl.get("forbidden_terms") or {}).get(ctx.tier, [])
    out: list[Violation] = []
    for term in forbidden:
        if term in ctx.text:
            out.append(
                Violation(
                    rule="forbidden_terms",
                    message=f"章型「{ctx.template}」({ctx.tier}) 禁用措辞「{term}」，属另一层次子风格，不得渗入。",
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
    forbidden = (tmpl.get("forbidden_subsection_terms") or {}).get(ctx.tier, [])
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
                            f"章型「{ctx.template}」({ctx.tier}) 禁设独立子项「{term}」"
                            "（该子项应并入对应子风格结构）。"
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
        if ctx.tier not in (entry.get("tiers") or []):
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
        if ctx.tier not in (entry.get("tiers") or []):
            continue
        if chap in required_in and not present:
            out.append(
                Violation(
                    rule="political_theory_missing",
                    message=(
                        f"归位章「{ctx.template}」({ctx.tier}) 须逐字含「{verbatim}」，"
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
    if ctx.tier not in (affective.get("tiers") or []):
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
                    f"归位章「{ctx.template}」({ctx.tier}) 须含价值导向/情感语"
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
        if ctx.tier not in (entry.get("tiers") or []):
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
    子串匹配（``in``）。``references`` 为空/None 则不校验（未补全事实依据时不误伤）。
    事实来源由调用方传入，非 SSoT。
    """
    if not ctx.references:
        return []
    out: list[Violation] = []
    for fact in ctx.references:
        if fact.value not in ctx.text:
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


@register
def _rule_fabrication(ctx: _LintContext) -> list[Violation]:
    """查臆造：正则从正文抽候选 token，须能在 references 找到依据。

    按类型（industry_code/credit/certificate）分别校验；仅当 references 含该类型
    ≥1 条时才校验该类型——调用方未提供某类型依据即视为该类型不在本章事实范围内，
    不误伤未补全 references 的章。token 抽取委托 ``extract_facts``，
    模式来自 SSoT ``fabrication``，避免校验器与指南漂移。
    """
    if not ctx.references:
        return []
    fab = ctx.cfg.get("fabrication") or {}
    by_type: dict[str, list[Fact]] = {}
    for f in ctx.references:
        by_type.setdefault(f.type, []).append(f)
    text_facts = extract_facts(ctx.text, ctx.cfg)
    out: list[Violation] = []

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


def lint(
    text: str,
    tier: str,
    *,
    style_guide_path: str | Path | None = None,
    domain: str | None = None,
    references: list[Fact] | None = None,
    materials: list[MaterialPayload] | None = None,
    hypotheses: list[HypothesisPayload] | None = None,
) -> list[Violation]:
    """对单章正文跑全部已注册规则，返回违规列表（纯函数）。

    ``style_guide_path`` 缺省用随包 ``style_guide.md``（不依赖工作目录）；
    显式传路径便于测试与替换指南。
    ``domain`` 标注被校验文本所属领域（如「金融」；通用为 None），
    供意识形态 (A) 的领域专属政治语双向校验。
    ``references`` 为本章调用方传入的事实依据（``Fact`` 列表），供「引用事实在位 +
    查臆造」校验；为空/None 则跳过该两条规则（不误伤未补全事实依据的章）。
    ``materials`` 为本章素材池（``MaterialPayload`` 列表），``hypotheses`` 为本章假说列表，
    供素材相关结构校验（无杜撰角标 / 池内 id 唯一 / hypothesis_id 合法 / 照抄型派生未标）；
    为空/None 则跳过素材相关规则（不误伤无素材语料）。
    """
    cfg = load_config(style_guide_path)
    normalized = normalize_cjk_ws(text)
    ctx = _LintContext(
        text=normalized,
        cfg=cfg,
        tier=tier,
        template=detect_chapter_template(normalized, cfg),
        domain=domain,
        references=references,
        materials=materials,
        hypotheses=hypotheses,
    )
    violations: list[Violation] = []
    for rule in _RULES:
        violations.extend(rule(ctx))
    return violations
