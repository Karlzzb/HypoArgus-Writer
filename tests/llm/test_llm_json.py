"""parse_json 的解析行为测试。"""

import pytest

from llm.llm_json import parse_json


def test_解析纯净的对象应答() -> None:
    assert parse_json('{"ok": true}', "步骤") == {"ok": True}


def test_剥掉代码围栏与前后文字() -> None:
    raw = '好的，结果如下：\n```json\n{"ok": true}\n```\n以上。'
    assert parse_json(raw, "步骤") == {"ok": True}


def test_容忍字符串值内的原始控制字符() -> None:
    """LLM 常在长文本字段里输出未转义的换行/制表符，解析必须容忍。"""
    raw = '{"reason": "第一行\n第二行\t缩进"}'
    assert parse_json(raw, "步骤") == {"reason": "第一行\n第二行\t缩进"}


def test_找不到_JSON_时报含步骤名的错() -> None:
    with pytest.raises(ValueError, match="引文语义核查"):
        parse_json("完全没有结构化内容", "引文语义核查")


def test_JSON_残缺时报含步骤名的错() -> None:
    with pytest.raises(ValueError, match="不是合法 JSON"):
        parse_json('{"ok": ', "步骤")
