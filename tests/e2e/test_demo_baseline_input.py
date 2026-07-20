"""回归基准输入（issue #19 固化）的守护测试。

真实 E2E 复跑的输入素材固化在 `scripts/demo_task.baseline.json`，
demo.py 创建任务时以其为唯一输入来源。本测试守护基准内容不被无意改动：
若确需演进基准，须同步更新此处快照并说明回归基准语义变化。
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = REPO_ROOT / "scripts" / "demo_task.baseline.json"

EXPECTED_TASK = {
    "user_intent": (
        "按「人才培养方案总结（汇报）模版」，为智能网联汽车技术专业"
        "（460704）2025 级高职专科人才培养方案撰写一份评审汇报用的总结"
    ),
    "user_identity": "高职院校教务处教师",
    "session_id": "demo-session",
}


def test_baseline_fixture_exists_and_matches_snapshot() -> None:
    payload = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert payload["task"] == EXPECTED_TASK


def test_demo_loads_task_from_baseline_fixture() -> None:
    from scripts.demo import BASELINE_TASK_PATH, load_baseline_task

    assert BASELINE_TASK_PATH == BASELINE_PATH
    assert load_baseline_task() == EXPECTED_TASK
