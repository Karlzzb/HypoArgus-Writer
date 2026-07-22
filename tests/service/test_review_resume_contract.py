"""恢复值契约测试：服务层产地与人工中断点消费方必须同形（issue #49 扩展）。

恢复值契约横跨三处：HTTP 层 ReviewRequest（入参形状）、任务服务层
build_resume_value（恢复值唯一产地）、human_review_gate 节点校验（消费方）。
本测试锚定三处的动作全集与字段形状一致，防止任何一侧单独漂移——
无论解析走 FakeLLM 打桩还是真实 LLM，恢复值都由程序侧构造，形状同此契约。
"""

from typing import get_args

import pytest

from nodes.human_review_gate import RESUME_ACTIONS, _validate_decision
from service.app import ReviewRequest
from service.task_service import InvalidReview, build_resume_value


def test_动作全集三处一致() -> None:
    """HTTP 请求模型、服务层产地、节点校验方接受同一组动作。"""
    http_actions = set(get_args(ReviewRequest.model_fields["action"].annotation))
    assert http_actions == set(RESUME_ACTIONS)
    for action in RESUME_ACTIONS:
        feedback = "意见" if action == "revise" else None
        resume_value = build_resume_value(action, feedback)
        validated_action, _ = _validate_decision(resume_value, confirmable=True)
        assert validated_action == action


def test_finalize与confirm恢复值只携action字段() -> None:
    assert build_resume_value("finalize", None) == {"action": "finalize"}
    assert build_resume_value("confirm", None) == {"action": "confirm"}
    # confirm 忽略多余的 feedback，不把它带进恢复值。
    assert build_resume_value("confirm", "多余意见") == {"action": "confirm"}


def test_revise恢复值携裁剪后的意见文本() -> None:
    assert build_resume_value("revise", "  引言更简洁  ") == {
        "action": "revise",
        "feedback": "引言更简洁",
    }


@pytest.mark.parametrize("feedback", [None, "", "   "])
def test_revise缺意见_服务层与节点同判非法(feedback: str | None) -> None:
    with pytest.raises(InvalidReview):
        build_resume_value("revise", feedback)
    with pytest.raises(ValueError):
        _validate_decision({"action": "revise", "feedback": feedback}, confirmable=True)


def test_非法动作_服务层与节点同判非法() -> None:
    with pytest.raises(InvalidReview):
        build_resume_value("publish", None)
    with pytest.raises(ValueError):
        _validate_decision({"action": "publish"}, confirmable=True)


def test_confirm仅在待确认清单存在时被节点接受() -> None:
    resume_value = build_resume_value("confirm", None)
    with pytest.raises(ValueError, match="confirm 不可用"):
        _validate_decision(resume_value, confirmable=False)
    assert _validate_decision(resume_value, confirmable=True) == ("confirm", "")
