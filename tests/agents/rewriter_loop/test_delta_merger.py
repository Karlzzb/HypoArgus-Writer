"""DeltaMerger 单元测试：字符数阈值与时间窗口的合并行为。

合并粒度直接影响逐字流跨线程调度频次，须确定性可复现：FakeLLM 按 chunk_size
切分、各片段在同一 monotonic 瞬时到达，故单测里关掉时间窗口（``flush_ms=0``）
纯按字符数驱动；时间窗口分支用注入的假时钟确定性触发。
"""

from __future__ import annotations

from typing import Any

from agents.rewriter_loop.delta_merger import DeltaMerger


def _capture() -> tuple[list[tuple[str, str]], Any]:
    """构造捕获回调与帧列表：on_flush(kind, text) 入帧。"""
    frames: list[tuple[str, str]] = []
    return frames, (lambda kind, text: frames.append((kind, text)))


def test_字符数阈值_累积达标即flush且帧拼接为完整正文() -> None:
    """累积到 flush_chars 即 flush 该 kind；各帧 delta 拼接为全部正文。"""
    frames, on_flush = _capture()
    merger = DeltaMerger(on_flush, flush_chars=4, flush_ms=0)
    # 12 字符 / 阈值 4 → 至少 3 帧（末帧由 flush_remaining 兜底）。
    for ch in "abcdefghijkl":
        merger.feed("content", ch)
    merger.flush_remaining()
    content_frames = [text for kind, text in frames if kind == "content"]
    assert "".join(content_frames) == "abcdefghijkl"
    # 每帧不超过阈值（末帧由 flush_remaining 触发，可能不足阈值）。
    assert all(len(text) <= 4 for text in content_frames[:-1])
    assert len(content_frames) >= 3


def test_字符数阈值_空片段不触发任何帧() -> None:
    """空文本 feed 为 no-op，不进缓冲、不外发。"""
    frames, on_flush = _capture()
    merger = DeltaMerger(on_flush, flush_chars=4, flush_ms=0)
    merger.feed("content", "")
    merger.flush_remaining()
    assert frames == []


def test_时间窗口_超时未达字符数也flush() -> None:
    """时间窗口驱动：elapsed >= flush_ms 即 flush，即使未达字符数阈值。"""
    # 可变状态时钟：构造时 t=0，feed 推进 1ms，第三次 feed 前外部跳到 100ms。
    state = {"t": 0.0}

    def clock() -> float:
        return state["t"]

    frames, on_flush = _capture()
    # flush_ms=10：第二次 feed elapsed=1ms 不到；第三次前跳到 100ms 触发。
    merger = DeltaMerger(on_flush, flush_chars=100, flush_ms=10, clock=clock)
    state["t"] = 0.001
    merger.feed("content", "a")  # elapsed=1ms，未触发
    state["t"] = 0.001
    merger.feed("content", "b")  # elapsed=1ms（_last_flush_ts 仍 0），未触发
    assert [t for _, t in frames] == []
    state["t"] = 0.1
    merger.feed("content", "c")  # elapsed=100ms ≥ 10ms，触发 flush "abc"
    assert [t for _, t in frames] == ["abc"]


def test_时间窗口_设为0关闭时间分支只按字符数() -> None:
    """flush_ms=0 关闭时间窗口分支；即使时钟大量前进也只按字符数 flush。"""
    state = {"t": 0.0}

    def clock() -> float:
        return state["t"]

    frames, on_flush = _capture()
    merger = DeltaMerger(on_flush, flush_chars=10, flush_ms=0, clock=clock)
    state["t"] = 100.0
    merger.feed("content", "abc")  # 3 < 10，即便 elapsed 巨大也不 flush
    state["t"] = 100.0
    merger.feed("content", "def")  # 6 < 10
    assert frames == []
    state["t"] = 100.0
    merger.feed("content", "ghij")  # 10 ≥ 10 → flush
    assert [t for _, t in frames] == ["abcdefghij"]


def test_帧数随阈值反向变化_小阈值多帧大阈值少帧() -> None:
    """同一正文按不同阈值切出不同帧数：小阈值→多帧、大阈值→少帧。"""
    text = "abcdefghij"  # 10 字符

    frames_small, on_small = _capture()
    merger_small = DeltaMerger(on_small, flush_chars=2, flush_ms=0)
    for ch in text:
        merger_small.feed("content", ch)
    merger_small.flush_remaining()

    frames_large, on_large = _capture()
    merger_large = DeltaMerger(on_large, flush_chars=100, flush_ms=0)
    for ch in text:
        merger_large.feed("content", ch)
    merger_large.flush_remaining()

    small_count = len([t for _, t in frames_small])
    large_count = len([t for _, t in frames_large])
    assert small_count > large_count
    # 大阈值下全部正文由 flush_remaining 一次性 flush → 单帧。
    assert large_count == 1
    # 小阈值（2）下至少 5 帧。
    assert small_count >= 5


def test_flush_remaining_残余缓冲一次性外发() -> None:
    """流结束时调 flush_remaining 把残余缓冲 flush（正文先、思考后）。"""
    frames, on_flush = _capture()
    merger = DeltaMerger(on_flush, flush_chars=100, flush_ms=0)
    merger.feed("content", "正文残余")
    merger.feed("thinking", "推理残余")
    assert frames == []  # 阈值大未达，未 flush
    merger.flush_remaining()
    # flush_remaining 顺序：content 先、thinking 后。
    assert frames == [("content", "正文残余"), ("thinking", "推理残余")]


def test_两kind各自累积互不串扰() -> None:
    """content 与 thinking 各自缓冲、各自达阈值即 flush，不互相串字。"""
    frames, on_flush = _capture()
    merger = DeltaMerger(on_flush, flush_chars=3, flush_ms=0)
    merger.feed("content", "ab")
    merger.feed("thinking", "xy")
    merger.feed("content", "c")  # content 累积 3 → flush "abc"
    merger.feed("thinking", "z")  # thinking 累积 3 → flush "xyz"
    merger.flush_remaining()
    assert ("content", "abc") in frames
    assert ("thinking", "xyz") in frames
    content_text = "".join(t for k, t in frames if k == "content")
    thinking_text = "".join(t for k, t in frames if k == "thinking")
    assert content_text == "abc"
    assert thinking_text == "xyz"


def test_未知kind原样外发不进缓冲累积() -> None:
    """防御：未知 kind 不进缓冲累积，原样外发一次（保持向前兼容口径）。"""
    frames, on_flush = _capture()
    merger = DeltaMerger(on_flush, flush_chars=100, flush_ms=0)
    merger.feed("unknown_kind", "奇异片段")
    merger.flush_remaining()
    # 未知 kind 不进缓冲、flush_remaining 不发它（缓冲里没有），只 feed 时外发一次。
    assert frames == [("unknown_kind", "奇异片段")]
