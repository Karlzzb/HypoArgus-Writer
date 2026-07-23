"""增量 JSON 字段抽取器：从流式片段中增量抽出指定字符串字段的纯值。

writer 输出是 JSON-in-text（正文包在 ``chapter_text`` 字段里），逐字流需要把
生成中的 ``chapter_text`` 纯正文逐片段外发，而非原始 JSON 语法碎片。本模块实现
一个有限状态机：逐字符吃入流式片段，状态化地跳过前导噪声、定位目标键、
逐字解码目标值字符串（处理转义序列与跨 chunk 截断的 ``\\uXXXX``），并只外发
本片段新解码出的正文。

失败语义：遇到非法转义、非法 ``\\uXXXX`` 十六进制、目标键后跟非字符串值等
确定性语法错误，立即抛 ``FieldExtractionError``；流结束仍未完整抽出目标字段
（未闭合字符串 / 未找到目标键）同样在 ``finish`` 抛错——供上层触发退化重试。
"""

from __future__ import annotations

from collections.abc import Iterable

# JSON 字符串值的转义映射：单字符转义（``\\uXXXX`` 单独处理）。
_SIMPLE_ESCAPES: dict[str, str] = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


class FieldExtractionError(ValueError):
    """增量抽取中途失败：非法 JSON 或流结束前目标字段未完整抽出。"""


# 状态机帧类型（对象/数组上下文），用于判定字符串是键还是值。
_OBJ_WANT_KEY = "obj_want_key"  # 对象位置：期待一个键（``{`` 后或 ``,`` 后）
_OBJ_WANT_COLON = "obj_want_colon"  # 键字符串刚闭合，期待 ``:``
_OBJ_WANT_VALUE = "obj_want_value"  # ``:`` 后，期待一个值
_OBJ_AFTER_VALUE = "obj_after_value"  # 值刚闭合，期待 ``,`` 或 ``}``
_ARR_WANT_VALUE = "arr_want_value"  # 数组位置：期待一个值或 ``]``
_ARR_AFTER_VALUE = "arr_after_value"  # 值刚闭合，期待 ``,`` 或 ``]``
_SKIP_VALUE = "skip_value"  # 跳过非字符串的非目标值（数字/布尔/null）

# 非字符串值的起始字符（数字 / 布尔 / null）；目标字段值若非字符串即判失败。
_NON_STRING_VALUE_START = set("-0123456789tfn")


class JsonFieldExtractor:
    """逐字符吃入流式片段，增量抽出指定字符串字段的纯值。

    ``feed(chunk)`` 返回本片段新解码出的目标字段正文；目标字段完整闭合后
    进入终态，后续 ``feed`` 返回空串。``finish`` 在目标字段未完整抽出时抛
    ``FieldExtractionError``，成功则返回完整正文。

    仅处理对象根的 JSON（writer 契约恒为对象）；支持字段前后存在其他键、
    字段值前后存在其他字符串/非字符串值。非目标值仅结构化跳过，不外发。
    """

    def __init__(self, field_name: str = "chapter_text") -> None:
        self._field_name = field_name
        # 栈：每个对象/数组层一个帧类型。空栈表示仍在前导噪声中未进入 JSON。
        self._stack: list[str] = []
        self._state = "preamble"
        # 当前字符串片段的性质：键 / 目标值 / 其他值。
        self._string_kind: str = ""  # "key" | "target" | "other"
        self._key_buffer: str = ""  # 键字符串的解码累积（用于比对字段名）
        # ``\\u`` 转义：累积十六进制数字，凑满 4 位才解码。
        self._unicode_buffer: str = ""
        # 目标值正文：完整累积（finish 返回）与逐片段外发分别维护。
        self._value: str = ""
        self._awaiting_target_value = False  # 目标键已闭合，期待紧随的值字符串
        self._done = False

    def feed(self, chunk: str) -> str:
        """吃入一个流式片段，返回本片段新解码出的目标字段正文。"""
        if self._done:
            return ""
        emitted: list[str] = []
        for char in chunk:
            self._step(char, emitted)
            if self._done:
                break
        return "".join(emitted)

    def finish(self) -> str:
        """流结束：目标字段已完整抽出则返回完整正文，否则抛 ``FieldExtractionError``。"""
        if self._done:
            return self._value
        if (
            self._state in ("in_string", "escape", "unicode")
            and self._string_kind == "target"
        ):
            raise FieldExtractionError(
                f"流结束前目标字段「{self._field_name}」的字符串值未闭合"
            )
        if self._state in ("escape", "unicode"):
            raise FieldExtractionError("流结束前转义序列未完整")
        if self._state == "skip_value":
            raise FieldExtractionError("流结束前非字符串值未完整")
        raise FieldExtractionError(f"流结束前未完整抽出目标字段「{self._field_name}」")

    @property
    def value(self) -> str:
        """已抽出的目标字段正文（未完成时为部分值）。"""
        return self._value

    @property
    def done(self) -> bool:
        """目标字段是否已完整闭合。"""
        return self._done

    # --- 状态机单步推进 ---

    def _step(self, char: str, emitted: list[str]) -> None:
        state = self._state
        if state == "preamble":
            self._step_preamble(char)
        elif state == "structure":
            self._step_structure(char)
        elif state == "in_string":
            self._step_in_string(char, emitted)
        elif state == "escape":
            self._step_escape(char, emitted)
        elif state == "unicode":
            self._step_unicode(char, emitted)
        elif state == "skip_value":
            self._step_skip_value(char)
        else:  # pragma: no cover - 不可达
            raise FieldExtractionError(f"未知状态：{state}")

    def _step_preamble(self, char: str) -> None:
        # 前导噪声：跳到首个对象起始符 ``{``（writer 契约恒为对象）。
        if char == "{":
            self._stack.append(_OBJ_WANT_KEY)
            self._state = "structure"

    def _step_structure(self, char: str) -> None:
        # 根对象已闭合（目标未抽出）：忽略尾随噪声，finish 兜底报「未找到」。
        if not self._stack:
            return
        top = self._stack[-1]
        if char.isspace():
            return
        if char == "{":
            self._stack.append(_OBJ_WANT_KEY)
            return
        if char == "[":
            self._stack.append(_ARR_WANT_VALUE)
            self._awaiting_target_value = False  # 值不会直接是数组字符串
            return
        if char == "}":
            self._close_object(top)
            return
        if char == "]":
            if top not in (_ARR_WANT_VALUE, _ARR_AFTER_VALUE):
                raise FieldExtractionError(f"此处不应出现 ']'（栈顶={top}）")
            self._stack.pop()
            self._after_value()
            return
        if char == ",":
            self._on_comma(top)
            return
        if char == ":":
            if top != _OBJ_WANT_COLON:
                raise FieldExtractionError(f"此处不应出现 ':'（栈顶={top}）")
            self._stack[-1] = _OBJ_WANT_VALUE
            return
        if char == '"':
            self._start_string(top)
            return
        # 非字符串值起始（数字 / 布尔 / null）。
        if char in _NON_STRING_VALUE_START:
            self._start_non_string_value(top, char)
            return
        raise FieldExtractionError(f"JSON 结构中出现非法字符 {char!r}")

    def _close_object(self, top: str) -> None:
        if top not in (_OBJ_WANT_KEY, _OBJ_AFTER_VALUE):
            raise FieldExtractionError(f"此处不应出现 '}}'（栈顶={top}）")
        self._stack.pop()
        if self._stack:
            self._after_value()
        # 根对象闭合后栈空：留在结构态，由 _step_structure 的空栈守卫忽略尾随。

    def _on_comma(self, top: str) -> None:
        if top == _OBJ_AFTER_VALUE:
            self._stack[-1] = _OBJ_WANT_KEY
            self._awaiting_target_value = False
            return
        if top == _ARR_AFTER_VALUE:
            self._stack[-1] = _ARR_WANT_VALUE
            return
        raise FieldExtractionError(f"此处不应出现 ','（栈顶={top}）")

    def _start_string(self, top: str) -> None:
        if top == _OBJ_WANT_KEY:
            self._string_kind = "key"
            self._key_buffer = ""
        elif top == _OBJ_WANT_VALUE:
            if self._awaiting_target_value:
                self._string_kind = "target"
                self._awaiting_target_value = False
            else:
                self._string_kind = "other"
        elif top in (_ARR_WANT_VALUE, _ARR_AFTER_VALUE):
            # 数组元素中的字符串不是目标字段（chapter_text 必为对象顶层字符串）。
            self._string_kind = "other"
        else:
            raise FieldExtractionError(f"此处不应开始字符串（栈顶={top}）")
        self._state = "in_string"

    def _start_non_string_value(self, top: str, char: str) -> None:
        if top == _OBJ_WANT_VALUE and self._awaiting_target_value:
            raise FieldExtractionError(f"目标字段「{self._field_name}」的值不是字符串")
        self._awaiting_target_value = False
        self._state = "skip_value"

    def _step_in_string(self, char: str, emitted: list[str]) -> None:
        if char == '"':
            self._close_string()
            return
        if char == "\\":
            self._state = "escape"
            return
        # 原始控制字符（未转义的换行等）按 JSON 严格语义为非法；但既有
        # parse_json 以 strict=False 容忍，本抽取器同样放行，原样外发，
        # 与非流式解析口径一致，避免真实模型偶发控制字符打断逐字流。
        self._emit_string_char(char, emitted)

    def _close_string(self) -> None:
        kind = self._string_kind
        self._string_kind = ""
        self._state = "structure"
        if kind == "key":
            if self._key_buffer == self._field_name:
                self._awaiting_target_value = True
            self._key_buffer = ""
            # 键闭合后栈顶应从 _OBJ_WANT_KEY 推进到 _OBJ_WANT_COLON。
            if self._stack[-1] == _OBJ_WANT_KEY:
                self._stack[-1] = _OBJ_WANT_COLON
        elif kind == "target":
            self._done = True
            # 目标值字符串闭合：根上不必继续解析后续键。
        else:  # other
            self._after_value()

    def _after_value(self) -> None:
        top = self._stack[-1]
        if top == _OBJ_WANT_VALUE:
            self._stack[-1] = _OBJ_AFTER_VALUE
        elif top == _ARR_WANT_VALUE:
            self._stack[-1] = _ARR_AFTER_VALUE
        # 其余帧类型（对象/数组闭合后的父层）保持不变。

    def _emit_string_char(self, char: str, emitted: list[str]) -> None:
        kind = self._string_kind
        if kind == "key":
            self._key_buffer += char
        elif kind == "target":
            self._value += char
            emitted.append(char)
        # other：丢弃。

    def _step_escape(self, char: str, emitted: list[str]) -> None:
        if char == "u":
            self._unicode_buffer = ""
            self._state = "unicode"
            return
        if char in _SIMPLE_ESCAPES:
            decoded = _SIMPLE_ESCAPES[char]
        else:
            raise FieldExtractionError(f"非法转义序列：\\{char}")
        self._emit_string_char(decoded, emitted)
        self._state = "in_string"

    def _step_unicode(self, char: str, emitted: list[str]) -> None:
        if char not in "0123456789abcdefABCDEF":
            raise FieldExtractionError(f"非法 \\uXXXX 转义：{char!r} 不是十六进制数字")
        self._unicode_buffer += char
        if len(self._unicode_buffer) < 4:
            return
        decoded = chr(int(self._unicode_buffer, 16))
        self._unicode_buffer = ""
        self._state = "in_string"
        self._emit_string_char(decoded, emitted)

    def _step_skip_value(self, char: str) -> None:
        # 跳过数字 / 布尔 / null：以结构分隔符（``,}``]``）或空白为界收尾，
        # 随后把分隔符交还结构态正常推进（``,}``]`` 走逗号/闭合逻辑）。
        if char.isspace():
            self._after_value()
            self._state = "structure"
            return
        if char in ",}]":
            self._after_value()
            self._state = "structure"
            self._step_structure(char)
            return
        # 其余字符（数字、字母、小数点、正负号、e/E）属值 token 内部，继续吞。


def feed_all(extractor: JsonFieldExtractor, chunks: Iterable[str]) -> str:
    """便利函数：把多片段依次喂入抽取器，返回完整目标字段正文。

    逐片段喂入后调用 ``finish`` 校验完整性（不完整或非法则抛
    ``FieldExtractionError``）；仅供单测与 mock 档拼接复用，逐字流真链路
    逐片段外发，不走本函数。
    """
    for chunk in chunks:
        extractor.feed(chunk)
    return extractor.finish()
