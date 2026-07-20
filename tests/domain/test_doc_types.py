"""文种注册表纯函数单测：四模板映射、变体承载、无命中与未登记兑底。"""

from domain.doc_types import (
    DOC_TYPE_REGISTRY,
    GENERIC_DOC_TYPE,
    resolve_doc_type,
)


def test_注册表_登记现有四模板() -> None:
    assert set(DOC_TYPE_REGISTRY) == {
        "本科职业教育人才培养方案模版.md",
        "高职专科人才培养方案模版.md",
        "人才培养方案总结（汇报）模版.md",
        "学院级多专业培养方案模版.md",
    }


def test_解析_人培两模板映射两变体() -> None:
    assert resolve_doc_type("本科职业教育人才培养方案模版.md") == ("人才培养方案", "本科")
    assert resolve_doc_type("高职专科人才培养方案模版.md") == ("人才培养方案", "高职")


def test_解析_汇报模板映射汇报材料无变体() -> None:
    assert resolve_doc_type("人才培养方案总结（汇报）模版.md") == ("汇报材料", None)


def test_解析_学院级多专业映射人培无变体() -> None:
    assert resolve_doc_type("学院级多专业培养方案模版.md") == ("人才培养方案", None)


def test_解析_自由结构无模板_落通用公文兑底() -> None:
    assert resolve_doc_type(None) == (GENERIC_DOC_TYPE, None)


def test_解析_未登记模板_同样落通用公文兑底() -> None:
    assert resolve_doc_type("某个未登记的模版.md") == (GENERIC_DOC_TYPE, None)
