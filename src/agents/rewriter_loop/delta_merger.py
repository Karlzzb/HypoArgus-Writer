"""逐字流增量合并器：在跨线程边界（EventHook 发布）之前先在工作线程上按帧合并。

逐字流原始片段可能逐 token 极细（真实 LLM 单字符增量常见），若每个 token 都跨线程
调度一次 EventHub.publish，会在事件循环线程上造成巨量 call_soon_threadsafe 调度，
挤压图运行。本合并器把同一 kind 的累积正文/推理片段攒到「字符数阈值」或
「时间窗口」再一次性经 ``on_flush(kind, text)`` 外发，降低跨线程调度频次。

设计要点：
- 字符数阈值（``flush_chars``）是主驱动：累积到阈值即 flush 该 kind 的缓冲并复位；
  FakeLLM 按 ``chunk_size`` 确定性切分、各片段在同一 monotonic 瞬时到达，故
  单测里关掉时间窗口（``flush_ms=0``）即可按字符数确定性驱动。
- 时间窗口（``flush_ms``）只对慢速真实流有意义：真实模型逐 token 间隔可达数十毫秒，
  字符数未到但已超时间窗口也须 flush，避免前端长时间看不到逐字推进；
  ``flush_ms=0`` 关闭时间窗口分支。
- 合并只发生在一个 attempt 内：attempt 切换时上层新建合并器，旧缓冲随 attempt
  丢弃（与「更高 attempt 调用方须丢弃旧增量、从零重建」的契约一致）。
- ``flush_remaining`` 在流正常结束时把两 kind 的残余缓冲一次性 flush（正文先、
  推理后），保证末尾片段不丢；错误路径不调 flush_remaining，残余片段随 attempt
  丢弃（已 flush 的帧仍带本 attempt 号、调用方在更高 attempt 丢弃重建）。
"""

from __future__ import annotations

import time
from collections.abc import Callable

# kind 取值对齐 CONTEXT.md 逐字流词汇表「按 kind 区分（content / thinking）」。
_CONTENT = "content"
_THINKING = "thinking"
_KINDS = (_CONTENT, _THINKING)


class DeltaMerger:
    """按字符数阈值或时间窗口合并逐字流片段，跨线程发布前先在工作线程上攒帧。

    ``on_flush(kind, text)`` 在每次攒够一帧时被调用（同步、调用方线程）；
    ``flush_chars`` 与 ``flush_ms`` 任一满足即 flush 当前 kind 的累积缓冲。
    时间窗口靠 ``clock`` 注入可确定性单测：FakeLLM 片段在同一 monotonic 瞬时
    到达，单测里设 ``flush_ms=0`` 关掉时间窗口分支、纯按字符数驱动。
    """

    def __init__(
        self,
        on_flush: Callable[[str, str], None],
        *,
        flush_chars: int,
        flush_ms: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._on_flush = on_flush
        self._flush_chars = flush_chars
        self._flush_ms = flush_ms
        self._clock = clock
        # 按 kind 维护独立缓冲：content 与 thinking 各自攒帧、各自判定阈值。
        self._buffers: dict[str, list[str]] = {kind: [] for kind in _KINDS}
        self._counts: dict[str, int] = {kind: 0 for kind in _KINDS}
        # 上次任意一次 flush 的时刻：时间窗口自上次 flush 起计，跨 kind 共享。
        self._last_flush_ts: float = self._clock()

    def feed(self, kind: str, text: str) -> None:
        """吃入一个片段：累积到对应 kind 缓冲，达标或超时即 flush 该 kind。"""
        if not text:
            return
        if kind not in self._buffers:
            # 防御：未知 kind 原样外发一次，不进缓冲累积（保持向前兼容口径）。
            self._on_flush(kind, text)
            return
        self._buffers[kind].append(text)
        self._counts[kind] += len(text)
        now = self._clock()
        elapsed_ms = (now - self._last_flush_ts) * 1000
        if self._counts[kind] >= self._flush_chars or (
            self._flush_ms > 0 and elapsed_ms >= self._flush_ms
        ):
            self._flush(kind)

    def flush_remaining(self) -> None:
        """流正常结束时把两 kind 的残余缓冲一次性 flush（正文先、推理后）。"""
        for kind in _KINDS:
            if self._counts[kind] > 0:
                self._flush(kind)

    def _flush(self, kind: str) -> None:
        """拼合该 kind 缓冲并经 on_flush 外发，复位该 kind 累积与时间戳。"""
        buffer = self._buffers[kind]
        if not buffer:
            return
        text = "".join(buffer)
        buffer.clear()
        self._counts[kind] = 0
        self._last_flush_ts = self._clock()
        self._on_flush(kind, text)
