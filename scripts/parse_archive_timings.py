#!/usr/bin/env python
"""解析构建过程档案，输出逐超步 / 逐单元墙钟，作为性能基线回路。

档案由 scripts/demo.py 落盘（var/demo_archive/<thread_id>.md），
逐条记录 graph_event 通道事件：行首序号、ISO 时间戳、事件类型、单元、
载荷、事件 id、父事件 id。node_end 的 parent 指向其 node_start 的 id，
据此配对算逐节点实例墙钟；HITL 边界取首个人工门 node_start 时间，
此前累加为 HITL 前工作量。

用法：
    python scripts/parse_archive_timings.py [ARCHIVE.md]
缺省取 var/demo_archive 下最新 .md（不含 -article）。

输出：逐节点实例表 + 逐单元汇总 + 阶段口径（HITL 前 / 写作段）。
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = REPO_ROOT / "var" / "demo_archive"

# 行示例：
#   882. `2026-07-23T07:10:41.179459+00:00` **node_start** unit=`chapter_drafter` payload=`{...}` id=`ab47549e` parent=`9d33a1f1`
_EVENT_RE = re.compile(
    r"^\s*\d+\.\s+"                           # 行首序号
    r"`(?P<ts>[^`]+)`\s+"                     # ISO 时间戳
    r"\*\*(?P<type>[a-z_]+)\*\*\s+"            # 事件类型
    r"unit=`(?P<unit>[^`]+)`"                 # 单元
    r"(?:\s+payload=`(?P<payload>.*?)`)"      # 载荷（非贪婪，到反引号）
    r"\s+id=`(?P<id>[^`]+)`"                  # 事件 id
    r"\s+parent=`(?P<parent>[^`]+)`"          # 父事件 id
)


def _parse_ts(ts: str) -> datetime:
    """ISO8601 带时区时间戳解析为 aware datetime。"""
    return datetime.fromisoformat(ts)


def _parse_payload(raw: str | None) -> dict:
    """载荷是 JSON 文本，解析失败兜底空 dict。"""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def parse_events(path: Path) -> list[dict]:
    """从档案抽取全部事件行，返回事件字典列表（按出现顺序）。"""
    events: list[dict] = []
    in_event_section = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if "完整事件流" in line:
            in_event_section = True
        if not in_event_section:
            continue
        m = _EVENT_RE.match(line)
        if not m:
            continue
        d = m.groupdict()
        payload = _parse_payload(d["payload"])
        events.append(
            {
                "ts": _parse_ts(d["ts"]),
                "type": d["type"],
                "unit": d["unit"],
                "id": d["id"],
                "parent": d["parent"],
                "step": payload.get("step"),
                "chapter_id": payload.get("chapter_id"),
                "interrupted": payload.get("interrupted", False),
            }
        )
    return events


def pair_node_instances(events: list[dict]) -> list[dict]:
    """node_end 经 parent 指向 node_start 的 id，配对算逐实例墙钟。"""
    starts = {e["id"]: e for e in events if e["type"] == "node_start"}
    instances: list[dict] = []
    for e in events:
        if e["type"] != "node_end":
            continue
        start = starts.get(e["parent"])
        if start is None:
            continue
        dur = (e["ts"] - start["ts"]).total_seconds()
        instances.append(
            {
                "unit": e["unit"],
                "step": e["step"],
                "chapter_id": start["chapter_id"] or e["chapter_id"],
                "start": start["ts"],
                "end": e["ts"],
                "dur": dur,
            }
        )
    return instances


def _fmt_ts(dt: datetime) -> str:
    return dt.strftime("%H:%M:%S")


def print_per_instance(instances: list[dict]) -> None:
    print("逐节点实例（按开始时间）：")
    print(f"{'step':>4}  {'unit':<22} {'chapter':<8} {'start':>10} {'end':>10} {'dur_s':>8}")
    for ins in sorted(instances, key=lambda x: x["start"]):
        ch = ins["chapter_id"] or "-"
        print(
            f"{str(ins['step']):>4}  {ins['unit']:<22} {ch:<8} "
            f"{_fmt_ts(ins['start']):>10} {_fmt_ts(ins['end']):>10} {ins['dur']:>8.1f}"
        )


def print_per_unit(instances: list[dict], total_wall: float) -> None:
    """逐单元汇总：agent_time = 各实例时长累加（并行重叠会计入多次，
    非墙钟）；反映该单元累计 LLM 工作量，墙钟看阶段口径。"""
    print("\n逐单元汇总（agent_time = 实例时长累加，含并行重叠）：")
    by_unit: dict[str, list[float]] = {}
    for ins in instances:
        by_unit.setdefault(ins["unit"], []).append(ins["dur"])
    print(f"{'unit':<22} {'count':>5} {'agent_s':>9}")
    for unit in sorted(by_unit, key=lambda u: -sum(by_unit[u])):
        durs = by_unit[unit]
        print(f"{unit:<22} {len(durs):>5} {sum(durs):>9.1f}")


def _hitl_boundary(instances: list[dict]) -> datetime | None:
    """首个人工门 node_start 时间即 HITL 边界。"""
    for ins in sorted(instances, key=lambda x: x["start"]):
        if ins["unit"] == "human_review_gate":
            return ins["start"]
    return None


def _span(items: list[dict]) -> float:
    """阶段墙钟 = max(end) - min(start)，正确反映并行扇出受最慢分支约束。"""
    if not items:
        return 0.0
    return (max(i["end"] for i in items) - min(i["start"] for i in items)).total_seconds()


def print_stages(instances: list[dict], total_wall: float) -> None:
    """按 issue #64 口径分段：HITL 前 / 首写(stage3) / 回退轮1(stage4)。
    阶段墙钟用 span（并行扇出受最慢分支约束），非实例时长累加。"""
    boundary = _hitl_boundary(instances)
    print("\n阶段口径（墙钟 = 阶段内 max(end)-min(start)）：")
    if total_wall:
        print(f"  全程墙钟（首事件→末事件）：{total_wall:.1f}s")
    if boundary is None:
        print("  （未发现人工门事件，无法划 HITL 边界）")
        return
    pre = [i for i in instances if i["end"] <= boundary]
    draft = [i for i in pre if i["unit"] == "chapter_drafter"]
    review = [i for i in pre if i["unit"] == "document_reviewer"]
    write = [i for i in pre if i["unit"] == "writing_orchestrator"]
    framework = [i for i in pre if i["unit"] == "framework_orchestrator"]
    search = [i for i in pre if i["unit"] == "reference_orchestrator"]
    stage4 = review + write
    for label, items in [
        ("stage1 framework", framework),
        ("stage2 search", search),
        ("stage3 首写 chapter_drafter", draft),
        ("stage4 回退轮1 review+write", stage4),
    ]:
        print(f"    {label:<34} {_span(items):>8.1f}s  ({len(items)} 实例)")
    print(f"  → HITL 前 stage3+4 写作段墙钟：{_span(draft + stage4):.1f}s")


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        path = Path(argv[1])
    else:
        candidates = sorted(
            (p for p in ARCHIVE_DIR.glob("*.md") if "-article" not in p.name),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            print(f"未在 {ARCHIVE_DIR} 找到档案", file=sys.stderr)
            return 1
        path = candidates[-1]
    if not path.exists():
        print(f"档案不存在：{path}", file=sys.stderr)
        return 1
    print(f"档案：{path}\n")

    events = parse_events(path)
    if not events:
        print("未解析到事件（档案格式不符？）", file=sys.stderr)
        return 1
    instances = pair_node_instances(events)
    total_wall = (events[-1]["ts"] - events[0]["ts"]).total_seconds()

    print_per_instance(instances)
    print_per_unit(instances, total_wall)
    print_stages(instances, total_wall)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
