"""调研报告验收基准输入（issue #28 固化）的守护测试。

验收基准按文种分目录（ADR-0005）：人培（汇报）基准保留
`scripts/demo_task.baseline.json` 原路径零改动，调研报告基准固化在
`scripts/baselines/调研报告/demo_task.baseline.json`，
demo.py 经 --task 指定后以其为唯一输入来源。
本测试守护基准内容不被无意改动：若确需演进基准，须同步更新此处快照
并说明回归基准语义变化。
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = REPO_ROOT / "scripts" / "baselines" / "调研报告" / "demo_task.baseline.json"

EXPECTED_TASK = {
    "user_intent": (
        "按「调研报告模版」，为云江职业技术学院智能网联汽车技术专业"
        "（460704）2025 届高职专科毕业生就业质量撰写一份年度监测调研报告"
        "（2025 年）"
    ),
    "user_identity": "高职院校质量管理办公室教师",
    "session_id": "demo-session-survey",
}


def test_调研报告基准存在且与快照一致() -> None:
    payload = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert payload["task"] == EXPECTED_TASK


def test_基准意图显式点名调研报告模版_保证注册表锚定文种() -> None:
    # 文种由模板选择经注册表确定性锚定（ADR-0005）：
    # 意图必须显式点名「调研报告模版」，品类识别方能稳定选中该模板。
    assert "调研报告模版" in EXPECTED_TASK["user_intent"]


def test_demo_可从指定路径加载调研报告基准() -> None:
    from scripts.demo import load_baseline_task

    assert load_baseline_task(BASELINE_PATH) == EXPECTED_TASK
