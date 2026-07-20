"""文种注册表：模板文件到 {文种, 变体} 的确定性映射（ADR-0005）。

文种不由模型自由裁量：品类识别选中模板文件后，经本表查得文种与变体，
由 framework_orchestrator 显式写入 State 的 doc_type/doc_variant 字段，
全链路只读、不可中途切换。
自由结构模式（无模板命中）与未登记模板一律落入「通用公文」兑底文种——
兑底只启用跨文种通用规则，是最保守的确定性选择。

新增文种 = 加模板 + 在此登记 + 加风格指南，三处闭合，不再动架构。
变体是文种内部的规则分支（目前仅人才培养方案声明本科/高职两个变体，
承接原 tier 概念）；无变体的条目取 None。
"""

from collections.abc import Mapping
from typing import Any

GENERIC_DOC_TYPE = "通用公文"
"""兑底文种：无模板命中或模板未登记时的确定性归宿。"""

_DEFAULT_TIER = "本科"

# 即层次的变体值：人才培养方案的两个变体承接原 tier 概念（ADR-0005）。
_TIER_VARIANTS = frozenset({"本科", "高职"})

DOC_TYPE_REGISTRY: dict[str, tuple[str, str | None]] = {
    "本科职业教育人才培养方案模版.md": ("人才培养方案", "本科"),
    "高职专科人才培养方案模版.md": ("人才培养方案", "高职"),
    "人才培养方案总结（汇报）模版.md": ("汇报材料", None),
    "学院级多专业培养方案模版.md": ("人才培养方案", None),
    "调研报告模版.md": ("调研报告", None),
}
"""模板文件名（docs_templates/ 下裸文件名）→ (文种, 变体)。"""


def resolve_doc_type(template_file: str | None) -> tuple[str, str | None]:
    """由模板文件名确定性解析 (文种, 变体)；无命中或未登记落通用公文。"""
    if template_file is None:
        return (GENERIC_DOC_TYPE, None)
    return DOC_TYPE_REGISTRY.get(template_file, (GENERIC_DOC_TYPE, None))


def carried_doc_facts(source: Mapping[str, Any]) -> tuple[str, str | None]:
    """从 State 或任务包读取携带的 (文种, 变体)：字段缺失或为空落通用公文兑底。

    兑底口径收敛于此，供节点、编排层、提示词适配器与打桩共用，避免多处同形漂移；
    兼容注册表落地前的旧存档（无文种字段的 State 与任务包）。
    """
    return (source.get("doc_type") or GENERIC_DOC_TYPE, source.get("doc_variant"))


def tier_from_variant(doc_variant: str | None) -> str:
    """人培文种内变体分键规则的取值兑底，与提示词「层次」行同源。

    人才培养方案的 本科/高职 变体即层次；无变体（含其他文种）回落缺省「本科」，
    保 学院级多专业(人培, None) 等既有链路产出零回归。变体分键规则
    （required_terms / forbidden_terms / glossary.tiers 等）仅存活于人培文种的
    合并配置内，其他文种无此类规则，该兑底值对其天然惰性。
    """
    if doc_variant in _TIER_VARIANTS:
        return doc_variant
    return _DEFAULT_TIER
