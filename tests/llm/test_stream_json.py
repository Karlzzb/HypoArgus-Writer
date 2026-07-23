"""增量 JSON 字段抽取器的状态机与边界测试。

覆盖验收四项（普通转义、``\\uXXXX`` 跨 chunk 截断、字段前后其他键、片段边界
恰落在字段引号处）与失败信号（非法 JSON 触发 ``FieldExtractionError`` 供退化重试）。
"""

from __future__ import annotations

import json

import pytest

from llm.stream_json import FieldExtractionError, JsonFieldExtractor, feed_all


def _chunks(text: str, size: int) -> list[str]:
    """按定长切片，模拟真实流式分块边界。"""
    return [text[i : i + size] for i in range(0, len(text), size)]


def test_普通转义按片段正确解码() -> None:
    """``\\n`` ``\\"`` ``\\\\`` 等普通转义须解码为对应字符，逐片段外发。"""
    # JSON 文本：chapter_text 值含 \n \" \\ 三种普通转义。
    raw = '{"chapter_text": "标题\\n正文\\"引用\\\\", "chapter_summary": "摘要"}'
    expected = json.loads(raw)["chapter_text"]
    assert expected == '标题\n正文"引用\\'

    ext = JsonFieldExtractor()
    deltas = [ext.feed(chunk) for chunk in _chunks(raw, 3)]
    assert "".join(deltas) == expected
    assert ext.finish() == expected


def test_uXXXX_跨chunk截断仍正确拼合() -> None:
    """``\\uXXXX`` 被切到多个片段时，须缓冲到四位凑齐再解码，不漏不乱。"""
    # 值 = 字 + 中(中) + A(A) + 尾
    raw = '{"chapter_text": "字\\u4e2d\\u0041尾"}'
    expected = json.loads(raw)["chapter_text"]
    assert expected == "字中A尾"

    # 用 size=2 强制 被切成多片，落在转义序列中间。
    ext = JsonFieldExtractor()
    deltas = [ext.feed(chunk) for chunk in _chunks(raw, 2)]
    assert "".join(deltas) == expected
    # 跨 chunk 缓冲期间该片段外发空串，凑齐后外发解码字符。
    assert "" in deltas
    assert "中" in "".join(deltas)
    assert "A" in "".join(deltas)


def test_字段前后有其他键只抽目标值() -> None:
    """目标字段前后存在其他字符串键时，只抽 chapter_text 纯值，不外发其他键值。"""
    raw = '{"prelude": "前置", "chapter_text": "正文[m1][m2]", "epilogue": "后置"}'
    ext = JsonFieldExtractor()
    deltas = [ext.feed(chunk) for chunk in _chunks(raw, 4)]
    assert "".join(deltas) == "正文[m1][m2]"
    # 其他键的值不得泄漏进 delta 流。
    assert "前置" not in deltas
    assert "后置" not in deltas


def test_字段前后有非字符串值也能跳过() -> None:
    """其他键的值是非字符串（数字/布尔/null）时，结构化跳过不打断抽取。"""
    raw = '{"count": 3, "flag": true, "chapter_text": "正文", "name": null}'
    assert feed_all(JsonFieldExtractor(), _chunks(raw, 5)) == "正文"


def test_片段边界恰落在字段开引号处() -> None:
    """片段边界正好在目标值开引号前后，须能正确进入值字符串捕获。"""
    # 边界 1：开引号在下一片段首字符。
    assert feed_all(JsonFieldExtractor(), ['{"chapter_text": ', '"hello"}']) == "hello"
    # 边界 2：开引号与首个内容分属不同片段。
    assert feed_all(JsonFieldExtractor(), ['{"chapter_text": "', 'hello"}']) == "hello"


def test_片段边界恰落在字段闭引号处() -> None:
    """片段边界正好在目标值闭引号处，须能正确收尾并停止。"""
    raw = '{"chapter_text": "hello"}'
    # 边界 1：闭引号前的内容在上一片段，闭引号独占下一片段首字符。
    assert feed_all(JsonFieldExtractor(), ['{"chapter_text": "hello', '"}']) == "hello"
    # 边界 2：逐字符喂入仍正确（最极端的边界）。
    assert feed_all(JsonFieldExtractor(), list(raw)) == "hello"


def test_前导噪声跳到首个对象起始符() -> None:
    """LLM 偶发输出围栏或导引文字，须跳到首个 ``{`` 后开始解析。"""
    raw = '好的，结果如下：\n```json\n{"chapter_text": "正文"}\n```'
    assert feed_all(JsonFieldExtractor(), _chunks(raw, 3)) == "正文"


def test_目标值含未转义控制字符照常外发() -> None:
    """与非流式 parse_json 的 strict=False 口径一致：容忍字符串值内控制字符。"""
    raw = '{"chapter_text": "第一行\n第二行\t缩进"}'
    assert feed_all(JsonFieldExtractor(), [raw]) == "第一行\n第二行\t缩进"


def test_非法转义中途抛失败信号() -> None:
    """非法转义 ``\\x`` 是确定性语法错误，feed 中途即抛，供上层退化重试。"""
    raw = '{"chapter_text": "前\\x后", "chapter_summary": "摘要"}'
    ext = JsonFieldExtractor()
    with pytest.raises(FieldExtractionError, match="非法转义序列"):
        ext.feed(raw)


def test_非法_uXXXX_抛失败信号() -> None:
    """``\\uXXXX`` 中非十六进制字符即抛明确失败信号。"""
    raw = '{"chapter_text": "前\\u12G4后"}'
    ext = JsonFieldExtractor()
    with pytest.raises(FieldExtractionError, match="不是十六进制数字"):
        ext.feed(raw)


def test_目标键后跟非字符串值抛失败信号() -> None:
    """目标字段值不是字符串（如数字）即判失败，供退化重试。"""
    raw = '{"chapter_text": 123, "chapter_summary": "摘要"}'
    ext = JsonFieldExtractor()
    with pytest.raises(FieldExtractionError, match="不是字符串"):
        ext.feed(raw)


def test_流结束前字符串未闭合抛失败信号() -> None:
    """流结束而目标值字符串未闭合，finish 抛明确失败信号。"""
    ext = JsonFieldExtractor()
    ext.feed('{"chapter_text": "未闭')
    with pytest.raises(FieldExtractionError, match="字符串值未闭合"):
        ext.finish()


def test_流结束前未找到目标键抛失败信号() -> None:
    """流结束未出现目标键，finish 抛明确失败信号，供退化重试。"""
    ext = JsonFieldExtractor()
    ext.feed('{"other": "x", "summary": "y"}')
    with pytest.raises(FieldExtractionError, match="未完整抽出目标字段"):
        ext.finish()


def test_跨chunk截断的反斜杠与u分离() -> None:
    """反斜杠落在片段尾、``u`` 落在下一片段首，仍须正确进入 unicode 转义。"""
    raw = '{"chapter_text": "前\\u4e2d后"}'
    # size=13 让反斜杠恰在某片段尾、u 在下片段首的边界可被触达。
    ext = JsonFieldExtractor()
    deltas = [ext.feed(c) for c in _chunks(raw, 7)]
    assert "".join(deltas) == json.loads(raw)["chapter_text"]


def test_自定义字段名抽取() -> None:
    """抽取器按字段名参数抽取，复用同一状态机抽 chapter_summary。"""
    raw = '{"chapter_text": "正文", "chapter_summary": "摘要"}'
    assert feed_all(JsonFieldExtractor("chapter_summary"), [raw]) == "摘要"


def test_done后后续feed返回空() -> None:
    """目标字段闭合后进入终态，后续片段不再外发，避免污染逐字流。"""
    raw = '{"chapter_text": "正文", "chapter_summary": "摘要"}'
    ext = JsonFieldExtractor()
    assert ext.feed(raw) == "正文"
    assert ext.done
    assert ext.feed("更多片段") == ""
    assert ext.finish() == "正文"
