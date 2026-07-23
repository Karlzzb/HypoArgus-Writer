#!/usr/bin/env python
"""单章检索智能体真实调用探针：端到端驱动 search_agent 真实现一次，
打印素材落库与诊断摘要，供 issue #52「检索返回为空」的秒级真实验证。

只跑一章、只跑一次真实检索（web/kb/structured 三通道 + 证据裁判），
不经过 framework/writing/review 全链路，把 issue #51 的复现从分钟级
真实 demo 降到秒级单智能体调用。

判定口径：
- materials 非空且 pass_count>0 → 检索链路恢复，issue #52 层 1 已修；
- materials 为空但 judge_integrity.judge_input_candidate_count>0 →
  候选已召回但未落库，查 mapping/verdict 折算或 integration_guard；
- call_counts.web_search>0 且 web_fetch=0 且 judge_input_candidate_count=0
  → 搜索通道仍空返回（凭据/端点/网络问题）。

用法：
    uv run python scripts/probe_search_agent.py
    uv run python scripts/probe_search_agent.py --task scripts/baselines/调研报告/demo_task.baseline.json
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env", override=False)

from agents.search_agent import make_search_agent  # noqa: E402
from domain.events import SUBAGENT_END  # noqa: E402


# 真实可检索的汽修章样例假说（与 test1 真实 demo 同域，保证 web 召回有内容）。
SAMPLE_TASK: dict[str, Any] = {
    "chapter_id": "probe-ch1",
    "points": [
        {"id": "p-1", "text": "新能源三电系统维修岗位能力要求与课程供给存在结构性缺口"},
    ],
    "hypotheses": [
        {
            "id": "h-1",
            "text": "头部车企初级维修技师岗位要求掌握电池管理系统故障诊断能力",
            "refute_condition": "若主流招聘 JD 与岗位标准未将 BMS 诊断列为必备技能则证伪",
        },
        {
            "id": "h-2",
            "text": "现行汽修专业人才培养方案中三电系统实训学时占比低于行业标准",
            "refute_condition": "若专业方案的三电实训占比已达头部车企标准 25% 至 30% 则证伪",
        },
        {
            "id": "h-3",
            "text": "新能源汽车渗透率突破 40% 后传统燃油车维修市场份额结构性收缩",
            "refute_condition": "若燃油车售后市场份额未随新能源渗透率上升而下降则证伪",
        },
    ],
    "genre": "行业白皮书",
    "existing_materials_digest": "",
}


def _load_task_from_baseline(path: Path) -> dict[str, Any]:
    """从文种基准任务构造单章检索任务包（取首章骨架派生可检索假说）。

    基准任务只含 user_intent 等驱动载荷，不含已展开的假说；此处退化为
    SAMPLE_TASK 形态仅当基准缺省。目前仅用 SAMPLE_TASK 做真实召回探测。
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    task = payload.get("task", payload)
    # 基准任务无假说结构，回退样例假说以保证检索通道有真实可检索文本。
    return {**SAMPLE_TASK, "chapter_id": "probe-baseline-ch1", "genre": task.get("genre", "行业白皮书")}


async def _probe(task: dict[str, Any]) -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    def hook(event_type: str, payload: dict[str, Any]) -> None:
        events.append((event_type, payload))

    adapter = make_search_agent(hook)
    print(f"model 单元：SEARCH_AGENT_LLM_MODEL={_env_or('SEARCH_AGENT_LLM_MODEL', '<缺省>')}")
    print(f"假说数：{len(task['hypotheses'])}，章：{task['chapter_id']}")
    print("-" * 60)

    t0 = time.perf_counter()
    result = await adapter.run(dict(task))
    elapsed = time.perf_counter() - t0

    materials = result.get("materials", [])
    by_verdict: dict[str, int] = {}
    for m in materials:
        by_verdict[m["verdict"]] = by_verdict.get(m["verdict"], 0) + 1

    print(f"耗时：{elapsed:.1f}s")
    print(f"materials 落库：{len(materials)} 条，verdict 分布：{by_verdict or '（空）'}")
    for m in materials[:5]:
        print(
            f"  - [{m['verdict']}] {m['source_kind']} {m.get('source')} "
            f"url={m.get('url')}"
        )
    if len(materials) > 5:
        print(f"  ... 其余 {len(materials) - 5} 条略")

    # 诊断摘要只在 subagent_end 事件里（适配层结束事件携带）。
    end_payloads = [p for t, p in events if t == SUBAGENT_END]
    diag = end_payloads[0].get("diagnostics", {}) if end_payloads else {}
    if diag:
        print("-" * 60)
        print("诊断摘要（来自 subagent_end）：")
        print(f"  total_elapsed_ms: {diag.get('total_elapsed_ms')}")
        print(f"  deadline_reached: {diag.get('deadline_reached')}")
        print(f"  call_counts: {diag.get('call_counts')}")
        print(f"  judge_integrity: {diag.get('judge_integrity')}")
        print(f"  gap_retrieval: {diag.get('gap_retrieval')}")
        if "weak_evidence_count" in diag:
            print(f"  weak_evidence_count: {diag['weak_evidence_count']}")
        if "pass_below_threshold" in diag:
            print(f"  pass_below_threshold: {diag['pass_below_threshold']}")

    print("-" * 60)
    _verdict(materials, diag)


def _verdict(materials: list[dict[str, Any]], diag: dict[str, Any]) -> None:
    call = diag.get("call_counts", {}) or {}
    judge = diag.get("judge_integrity", {}) or {}
    pass_count = sum(1 for m in materials if m["verdict"] == "pass")
    if materials and pass_count > 0:
        print(f"[绿] 检索恢复：{len(materials)} 条素材、{pass_count} 条 pass 落库。issue #52 层 1 已修。")
    elif materials:
        print(
            f"[黄] 召回到 {len(materials)} 条素材但 pass=0（全弱佐证/反例）。"
            "候选已入库但无强支撑——查裁判阈值或证据强度。"
        )
    elif judge.get("judge_input_candidate_count", 0) > 0:
        print(
            "[红] 候选已召回并送裁判但 materials=0：查 mapping 折算或"
            "integration_guard 是否在本项目集成路径截断（预期不截断）。"
        )
    elif call.get("web_search", 0) > 0 and call.get("web_fetch", 0) == 0:
        print(
            "[红] web_search 发起但 web_fetch=0 且候选=0：搜索通道仍空返回。"
            "查 VOLCANO_SEARCH_API_KEY / base_url / path / 端点契约。"
        )
    else:
        print("[红] materials=0 且无候选召回：检索通道未生效，查配置与凭据。")


def _env_or(name: str, default: str) -> str:
    import os

    return os.environ.get(name, default)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        metavar="PATH",
        type=Path,
        default=None,
        help="文种基准任务路径（目前仅取 genre，假说用内置样例）",
    )
    args = parser.parse_args()
    task = _load_task_from_baseline(args.task) if args.task else dict(SAMPLE_TASK)
    asyncio.run(_probe(task))


if __name__ == "__main__":
    main()
