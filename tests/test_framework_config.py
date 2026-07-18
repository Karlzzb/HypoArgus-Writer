"""论证体系数量上限配置读取的单元测试。"""

import pytest

from framework_config import FrameworkLimits, load_framework_limits


def test_未设置时全部取缺省值():
    limits = load_framework_limits({})
    assert limits == FrameworkLimits(
        max_points_per_chapter=4,
        max_hypotheses_per_point=3,
        max_hypotheses_total=60,
    )


def test_独立覆盖单个变量其余取缺省():
    limits = load_framework_limits({"FRAMEWORK_MAX_POINTS_PER_CHAPTER": "6"})
    assert limits.max_points_per_chapter == 6
    assert limits.max_hypotheses_per_point == 3
    assert limits.max_hypotheses_total == 60


def test_三个变量同时覆盖():
    env = {
        "FRAMEWORK_MAX_POINTS_PER_CHAPTER": "5",
        "FRAMEWORK_MAX_HYPOTHESES_PER_POINT": "2",
        "FRAMEWORK_MAX_HYPOTHESES_TOTAL": "40",
    }
    limits = load_framework_limits(env)
    assert limits == FrameworkLimits(
        max_points_per_chapter=5,
        max_hypotheses_per_point=2,
        max_hypotheses_total=40,
    )


def test_空字符串视为未配置回落缺省():
    env = {
        "FRAMEWORK_MAX_POINTS_PER_CHAPTER": "  ",
        "FRAMEWORK_MAX_HYPOTHESES_TOTAL": "",
    }
    limits = load_framework_limits(env)
    assert limits.max_points_per_chapter == 4
    assert limits.max_hypotheses_total == 60


@pytest.mark.parametrize("bad_value", ["abc", "3.5", "0", "-1"])
def test_非法值抛错并指明变量名(bad_value: str):
    env = {"FRAMEWORK_MAX_HYPOTHESES_PER_POINT": bad_value}
    with pytest.raises(ValueError, match="FRAMEWORK_MAX_HYPOTHESES_PER_POINT"):
        load_framework_limits(env)


def test_每个变量的非法值都指明各自变量名():
    with pytest.raises(ValueError, match="FRAMEWORK_MAX_POINTS_PER_CHAPTER"):
        load_framework_limits({"FRAMEWORK_MAX_POINTS_PER_CHAPTER": "x"})
    with pytest.raises(ValueError, match="FRAMEWORK_MAX_HYPOTHESES_TOTAL"):
        load_framework_limits({"FRAMEWORK_MAX_HYPOTHESES_TOTAL": "0"})
