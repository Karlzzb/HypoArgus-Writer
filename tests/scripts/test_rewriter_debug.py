"""rewriter_debug 调测脚本冒烟测试：空转模式经子进程整跑，断言关键输出与退出码。

只覆盖空转（FakeLLM）路径——--real 有真实花费，不进自动化测试。
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "rewriter_debug.py"
SAMPLE = REPO_ROOT / "scripts" / "rewriter_task.sample.json"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )


def test_空转整跑_事件流完整且产出最终结果() -> None:
    proc = _run()
    assert proc.returncode == 0, proc.stderr
    # 整跑走真编排（ADR-0006 T3 纯写作链路）：只发唯一一对写作调用事件，
    # 不再打印 lint_done / audit_done / revise_triggered（质检已移出 rewriter）。
    assert "llm_call_start" in proc.stdout and "llm_call_end" in proc.stdout
    for absent in ("lint_done", "audit_done", "revise_triggered"):
        assert absent not in proc.stdout
    assert "subagent_start" in proc.stdout and "subagent_end" in proc.stdout
    assert "最终产物（RewriteResult）" in proc.stdout
    assert "citations_ok" in proc.stdout


def test_step_write_只跑起草不进入后续环节() -> None:
    proc = _run("--step", "write")
    assert proc.returncode == 0, proc.stderr
    assert "起草（write）" in proc.stdout
    assert "校验（lint）" not in proc.stdout
    assert "自审（audit）" not in proc.stdout


def test_step_revise_fix_逐环节打印全部中间产物() -> None:
    proc = _run("--step", "revise-fix")
    assert proc.returncode == 0, proc.stderr
    for section in ("起草（write）", "校验（lint）", "自审（audit）", "修订（revise-fix）", "修订产物"):
        assert section in proc.stdout


def test_revise模式_读样例任务包并落实修订指令() -> None:
    proc = _run("--mode", "revise", "--task", str(SAMPLE))
    assert proc.returncode == 0, proc.stderr
    assert "修订落实：精简第一段" in proc.stdout
    assert '"mode": "revise"' in proc.stdout


def test_任务包缺字段_报可读错误退出(tmp_path: Path) -> None:
    bad = tmp_path / "bad_task.json"
    bad.write_text('{"mode": "draft"}', encoding="utf-8")
    proc = _run("--task", str(bad))
    assert proc.returncode != 0
    assert "缺少必备字段" in proc.stderr
