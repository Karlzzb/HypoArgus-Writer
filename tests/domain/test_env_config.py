"""整数环境变量共享读取逻辑的单元测试：正整数与非负整数两种口径。"""

import pytest

from domain.env_config import read_nonnegative_int, read_positive_int


def test_正整数_空值回落缺省():
    assert read_positive_int({}, "N", 3) == 3
    assert read_positive_int({"N": "  "}, "N", 3) == 3


def test_正整数_取值与越界():
    assert read_positive_int({"N": "5"}, "N", 3) == 5
    for bad in ("0", "-1", "x"):
        with pytest.raises(ValueError, match="N"):
            read_positive_int({"N": bad}, "N", 3)


def test_非负整数_空值回落缺省():
    assert read_nonnegative_int({}, "N", 1) == 1
    assert read_nonnegative_int({"N": ""}, "N", 1) == 1


def test_非负整数_允许零与越界():
    assert read_nonnegative_int({"N": "0"}, "N", 1) == 0
    assert read_nonnegative_int({"N": "4"}, "N", 1) == 4
    for bad in ("-1", "x"):
        with pytest.raises(ValueError, match="N"):
            read_nonnegative_int({"N": bad}, "N", 1)
