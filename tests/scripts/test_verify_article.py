"""verify_article 成品验收脚本单测：切章、按文种加载规则、变体随参数解析。

真实成品验收在 issue #28 真实 E2E 中执行；此处只用最小合成语料
覆盖脚本外部行为（CLI 参数、退出码、规则来源随 --doc-type 切换）。
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "verify_article.py"

from agents.rewriter_loop.style_linter import load_config  # noqa: E402

from scripts.verify_article import (  # noqa: E402
    quantitative_violations_strict,
    split_chapters,
)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )


def test_按二级标题切章且忽略围栏内的井号() -> None:
    text = "\n".join(
        [
            "## 一、监测概述",
            "正文甲。",
            "```",
            "## 这不是章标题",
            "```",
            "## 二、结论",
            "正文乙。",
        ]
    )
    chapters = split_chapters(text)
    assert [title for title, _ in chapters] == ["一、监测概述", "二、结论"]
    assert "这不是章标题" in chapters[0][1]


def test_量化断言复查_无同句角标即违规且变体随参数() -> None:
    cfg = load_config("调研报告")
    body = "## 一、监测概述与数据说明\n就业率较上年提升2.3个百分点。"
    violations = quantitative_violations_strict(body, cfg, None)
    assert violations, "无角标的量化断言应被判违规"
    marked = "## 一、监测概述与数据说明\n就业率较上年提升2.3个百分点[m1]。"
    assert quantitative_violations_strict(marked, cfg, None) == []
    # 变体参数经 tier_from_variant 兑底解析，非法值不抛错、行为同缺省。
    assert quantitative_violations_strict(marked, cfg, "高职") == []


def test_CLI_调研报告文种_情绪词与量化断言按文种规则判违规(tmp_path: Path) -> None:
    article = tmp_path / "sample-article.md"
    article.write_text(
        "## 一、监测概述与数据说明\n"
        "数据来源为全国高校毕业生就业管理系统，样本覆盖率达标[m1]。"
        "就业率令人振奋地较上年提升2.3个百分点。\n",
        encoding="utf-8",
    )
    proc = _run(str(article), "--doc-type", "调研报告", "--doc-variant", "")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "oral_blacklist" in proc.stdout
    assert "quantitative" in proc.stdout


def test_CLI_chapter_types_使维度章享受表章豁免(tmp_path: Path) -> None:
    # 观点标题的维度章：标题不命中章型模板，缺省失去表章豁免被字数下限误伤；
    # 经 --chapter-types 显式提供 State 章型后与生成期同口径、不误伤。
    article = tmp_path / "sample-article.md"
    article.write_text(
        "## 一、空间下沉：精准破局基层人才荒\n"
        "### （一）数据观察\n"
        "数据显示基层就业占比较上年提升2.3个百分点[m1]。\n"
        "| 区域 | 占比 |\n| --- | --- |\n| 基层 | 41% |\n",
        encoding="utf-8",
    )
    without = _run(str(article), "--doc-type", "调研报告", "--doc-variant", "")
    assert without.returncode == 1
    assert "word_count" in without.stdout
    with_types = _run(
        str(article),
        "--doc-type", "调研报告", "--doc-variant", "",
        "--chapter-types", "维度章",
    )
    assert with_types.returncode == 0, with_types.stdout + with_types.stderr


def test_CLI_chapter_types_数量不符时报可读错误(tmp_path: Path) -> None:
    article = tmp_path / "sample-article.md"
    article.write_text("## 一、监测概述与数据说明\n正文[m1]。\n", encoding="utf-8")
    proc = _run(
        str(article),
        "--doc-type", "调研报告",
        "--chapter-types", "维度章,维度章",
    )
    assert proc.returncode == 1
    assert "与切出的 1 章不符" in proc.stdout


def test_CLI_人培文种缺省参数可跑通并输出逐章报告(tmp_path: Path) -> None:
    article = tmp_path / "sample-article.md"
    article.write_text("## 一、培养目标\n本章阐述培养目标。\n", encoding="utf-8")
    proc = _run(str(article))
    assert "共 1 章" in proc.stdout
    assert "合计违规" in proc.stdout
