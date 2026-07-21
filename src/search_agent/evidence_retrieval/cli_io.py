"""Stable byte-level output helpers for the production CLI."""
from __future__ import annotations

import io
import json
import sys
from typing import Any, TextIO


def emit_public_json(payload: dict[str, Any], stream: TextIO | None = None) -> None:
    """Write one public-contract JSON object as UTF-8 on every platform.

    Windows can expose a GBK text wrapper even when stdout is redirected to a
    downstream service. Writing UTF-8 bytes to the underlying buffer keeps the
    CLI contract stable and prevents valid citation text from crashing the
    production runner. StringIO remains supported for tests and embedders.
    """
    stream = stream or sys.stdout
    text = json.dumps(payload, ensure_ascii=False)
    buffer = getattr(stream, "buffer", None)
    if isinstance(buffer, (io.BufferedIOBase, io.RawIOBase)) or hasattr(buffer, "write"):
        buffer.write(text.encode("utf-8") + b"\n")
        buffer.flush()
        return
    stream.write(text + "\n")
    stream.flush()
