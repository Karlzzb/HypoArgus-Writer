"""framework_orchestrator 节点单元测试：用假 LLM 预置 JSON 应答序列直接调用节点函数。

不跑全图，只验证节点的外部行为：品类识别与模板匹配、大纲生成（模板骨架实例化
与自由结构两条路径）、逐章论点、逐论点假说、ID 规则、占位标记、上限截断与筛选。
"""

import json
from pathlib import Path
from typing import Any

import pytest

from framework_config import FrameworkLimits
from framework_orchestrator import make_framework_orchestrator_node
from llm_client import FakeLLM
from state import WorkflowStatus, initial_state

LIMITS = FrameworkLimits(
    max_points_per_chapter=4, max_hypotheses_per_point=3, max_hypotheses_total=60
)


def _hyp(text: str = "假说", angle: str = "预言", retrievable: bool = True) -> dict[str, Any]:
    """构造一条合法的假说应答项。"""
    return {
        "text": text,
        "refute_condition": f"若检索到与「{text}」相反的公开证据则被证伪",
        "angle": angle,
        "evidence_retrievable": retrievable,
    }


@pytest.fixture()
def templates_dir(tmp_path: Path) -> Path:
    """构造一个只含单个小模板的临时模板目录。"""
    (tmp_path / "index.md").write_text(
        "模板\t典型调用场景\n测试模版.md\t人才培养方案撰写\n", encoding="utf-8"
    )
    (tmp_path / "测试模版.md").write_text(
        "# {专业名称}人才培养方案\n\n"
        "## 一、培养目标\n### （一）目标定位\n\n"
        "## 二、课程体系\n### （一）课程结构\n### （二）实践环节\n\n"
        "## 三、附则\n",
        encoding="utf-8",
    )
    return tmp_path


def _run_node(
    responses: list[Any],
    templates_dir: Path,
    limits: FrameworkLimits = LIMITS,
) -> tuple[dict[str, Any], FakeLLM]:
    """预置应答序列（dict/list 自动转 JSON 文本）后执行一次节点。"""
    fake = FakeLLM(
        [
            item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
            for item in responses
        ]
    )
    node = make_framework_orchestrator_node(
        lambda unit: fake, templates_dir=templates_dir, limits=limits
    )
    result = node(
        initial_state("写一篇软件工程专业人才培养方案", "专业撰稿人", "trace-fw")
    )
    return dict(result), fake


def test_模板路径_产出合规State且章节裁剪生效(templates_dir: Path) -> None:
    result, fake = _run_node(
        [
            {"genre": "人才培养方案", "template_file": "测试模版.md"},
            [
                {
                    "index": 1,
                    "applicable": True,
                    "title": "软件工程专业培养目标",
                    "subsections": ["目标定位"],
                },
                {
                    "index": 2,
                    "applicable": True,
                    "title": "软件工程课程体系",
                    "subsections": ["课程结构", "实践环节"],
                },
                {"index": 3, "applicable": False, "title": "附则", "subsections": []},
            ],
            [{"text": "论点甲"}],
            [_hyp("假说甲", angle="假设")],
            [{"text": "论点乙"}],
            [_hyp("假说乙", angle="失效模式")],
        ],
        templates_dir,
    )

    assert result["genre"] == "人才培养方案"
    assert result["template_id"] == "测试模版.md"
    assert result["status"] == WorkflowStatus.FRAMEWORK_BUILDING
    assert result["current_node_llm_config"]["unit"] == "framework_orchestrator"

    outline = result["outline"]
    # applicable=false 的第三章被裁剪，其余章节沿用模板骨架顺序。
    assert [chapter.title for chapter in outline] == [
        "软件工程专业培养目标",
        "软件工程课程体系",
    ]
    assert outline[1].subsections == ["课程结构", "实践环节"]
    # ID 规则由程序生成：章节 ch1…、论点 ch1-p1…、假说 ch1-p1-h1…。
    assert [chapter.id for chapter in outline] == ["ch1", "ch2"]
    assert outline[0].points[0].id == "ch1-p1"
    assert outline[0].points[0].text == "论点甲"
    assert outline[0].points[0].hypotheses[0].id == "ch1-p1-h1"
    assert outline[0].points[0].hypotheses[0].angle == "假设"
    assert outline[1].points[0].hypotheses[0].id == "ch2-p1-h1"
    # 每章各一次论点调用 + 每论点各一次假说调用 + 识别与大纲各一次。
    assert len(fake.calls) == 6


def test_自由结构路径_识别应答无模板时正常产出(templates_dir: Path) -> None:
    result, _ = _run_node(
        [
            {"genre": "行业评论", "template_file": None},
            [{"title": "引言", "subsections": ["背景"]}],
            [{"text": "论点"}],
            [_hyp()],
        ],
        templates_dir,
    )

    assert result["template_id"] is None
    assert result["genre"] == "行业评论"
    assert result["status"] == WorkflowStatus.FRAMEWORK_BUILDING
    outline = result["outline"]
    assert [chapter.id for chapter in outline] == ["ch1"]
    assert outline[0].subsections == ["背景"]
    assert outline[0].points[0].hypotheses[0].id == "ch1-p1-h1"


def test_识别应答给出不存在的模板文件_回落自由结构(templates_dir: Path) -> None:
    result, _ = _run_node(
        [
            {"genre": "未知", "template_file": "不存在的模版.md"},
            [{"title": "自由章节", "subsections": []}],
            [{"text": "论点"}],
            [_hyp()],
        ],
        templates_dir,
    )

    assert result["template_id"] is None
    assert result["outline"][0].title == "自由章节"


def test_残留填充变量被程序替换为待补充占位(templates_dir: Path) -> None:
    result, _ = _run_node(
        [
            # 围栏包裹的 JSON 也必须能解析。
            '```json\n{"genre": "方案", "template_file": null}\n```',
            [{"title": "{专业名称}的课程改革", "subsections": ["{专业名称}师资建设"]}],
            [{"text": "论点"}],
            [_hyp()],
        ],
        templates_dir,
    )

    chapter = result["outline"][0]
    assert chapter.title == "【待补充：专业名称】的课程改革"
    assert chapter.subsections == ["【待补充：专业名称】师资建设"]


def test_模板路径_变量缺失时标题含待补充占位且流程不阻塞(templates_dir: Path) -> None:
    result, _ = _run_node(
        [
            {"genre": "方案", "template_file": "测试模版.md"},
            # LLM 未代入 {专业名称}，标题残留原样变量 → 程序兜底替换为占位标记。
            [
                {
                    "index": 1,
                    "applicable": True,
                    "title": "{专业名称}培养目标",
                    "subsections": ["{专业名称}目标定位"],
                },
                {"index": 2, "applicable": False, "title": "", "subsections": []},
                {"index": 3, "applicable": False, "title": "", "subsections": []},
            ],
            [{"text": "论点"}],
            [_hyp()],
        ],
        templates_dir,
    )

    chapter = result["outline"][0]
    assert chapter.title == "【待补充：专业名称】培养目标"
    assert chapter.subsections == ["【待补充：专业名称】目标定位"]
    # 占位不阻塞：论点与假说照常生成。
    assert chapter.points[0].hypotheses[0].id == "ch1-p1-h1"


def test_模板路径_LLM缺失章节项时回落骨架标题并保留占位(templates_dir: Path) -> None:
    result, _ = _run_node(
        [
            {"genre": "方案", "template_file": "测试模版.md"},
            # 只返回第 1 章，第 2、3 章缺失 → 回落骨架标题与子标题，视为适用。
            [{"index": 1, "applicable": True, "title": "培养目标", "subsections": []}],
            [{"text": "论点1"}],
            [_hyp()],
            [{"text": "论点2"}],
            [_hyp()],
            [{"text": "论点3"}],
            [_hyp()],
        ],
        templates_dir,
    )

    outline = result["outline"]
    assert [chapter.title for chapter in outline] == ["培养目标", "课程体系", "附则"]
    assert outline[1].subsections == ["课程结构", "实践环节"]


def test_上限截断与总数配额耗尽后跳过假说调用(templates_dir: Path) -> None:
    small = FrameworkLimits(
        max_points_per_chapter=2, max_hypotheses_per_point=2, max_hypotheses_total=2
    )
    result, fake = _run_node(
        [
            {"genre": "方案", "template_file": None},
            [{"title": "唯一章", "subsections": []}],
            # 3 条论点 → 截断到 2 条。
            [{"text": "论点1"}, {"text": "论点2"}, {"text": "论点3"}],
            # 3 条假说 → 截断到 2 条，同时耗尽全文总数配额。
            [
                _hyp("假说1", angle="假设"),
                _hyp("假说2", angle="边界条件"),
                _hyp("假说3", angle="预言"),
            ],
        ],
        templates_dir,
        limits=small,
    )

    points = result["outline"][0].points
    assert [point.id for point in points] == ["ch1-p1", "ch1-p2"]
    assert [hyp.id for hyp in points[0].hypotheses] == ["ch1-p1-h1", "ch1-p1-h2"]
    # 配额耗尽：第二个论点不再发起假说调用，全程共 4 次 LLM 调用。
    assert points[1].hypotheses == []
    assert len(fake.calls) == 4


def test_证据不可检索与非法角度的假说被过滤(templates_dir: Path) -> None:
    result, _ = _run_node(
        [
            {"genre": "方案", "template_file": None},
            [{"title": "章", "subsections": []}],
            [{"text": "论点"}],
            [
                _hyp("不可检索", retrievable=False),
                {
                    "text": "非法角度",
                    "refute_condition": "条件",
                    "angle": "灵感",
                    "evidence_retrievable": True,
                },
                _hyp("合规假说", angle="竞争解释"),
            ],
        ],
        templates_dir,
    )

    hypotheses = result["outline"][0].points[0].hypotheses
    assert [hyp.text for hyp in hypotheses] == ["合规假说"]
    assert hypotheses[0].id == "ch1-p1-h1"


def test_应答不是合法JSON时抛ValueError并指明步骤(templates_dir: Path) -> None:
    with pytest.raises(ValueError, match="品类识别"):
        _run_node(["这不是 JSON"], templates_dir)
