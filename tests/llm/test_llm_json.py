"""parse_json 与 invoke_json 的解析、重试行为测试。"""

import pytest

from llm.llm_client import FakeLLM
from llm.llm_json import invoke_json, parse_json


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


def test_应答残缺时带纠错反馈重试一次成功() -> None:
    """真实 LLM 偶发输出残缺 JSON，须带纠错反馈重试而非整次运行失败。"""
    llm = FakeLLM(responses=['{"ok": ', '{"ok": true}'])
    assert invoke_json(llm, "步骤", "系统", "用户", dict) == {"ok": True}
    assert len(llm.calls) == 2
    # 重试消息须携带上次原始应答与纠错指令，帮助模型自我修正。
    retry_messages = llm.calls[1]
    assert retry_messages[:2] == llm.calls[0][:2]
    assert retry_messages[2] == {"role": "assistant", "content": '{"ok": '}
    assert retry_messages[3]["role"] == "user"
    assert "不是合法 JSON" in retry_messages[3]["content"]


def test_顶层类型不符同样重试() -> None:
    llm = FakeLLM(responses=['["数组"]', '{"ok": true}'])
    assert invoke_json(llm, "步骤", "系统", "用户", dict) == {"ok": True}
    assert len(llm.calls) == 2


def test_重试全部失败后抛含步骤名的错() -> None:
    llm = FakeLLM(responses=["坏1", "坏2", "坏3", "坏4"])
    with pytest.raises(ValueError, match="引文语义核查"):
        invoke_json(llm, "引文语义核查", "系统", "用户", dict)
    # 有界重试：共 3 次尝试后放弃，不无限消耗调用。
    assert len(llm.calls) == 3


def test_首次成功不触发重试() -> None:
    llm = FakeLLM(responses=['{"ok": true}'])
    assert invoke_json(llm, "步骤", "系统", "用户", dict) == {"ok": True}
    assert len(llm.calls) == 1
