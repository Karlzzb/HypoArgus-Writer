"""LLM 配置读取与回落逻辑的单元测试。"""

import pytest

from llm.llm_config import load_llm_config

GLOBAL_ENV = {
    "LLM_MODEL": "global-model",
    "LLM_BASE_URL": "https://global.example.com/v1",
    "LLM_API_KEY": "global-key",
}


def test_全部回落全局缺省():
    config = load_llm_config("framework_orchestrator", GLOBAL_ENV)
    assert config.model == "global-model"
    assert config.base_url == "https://global.example.com/v1"
    assert config.api_key == "global-key"


def test_前缀变量优先生效():
    env = GLOBAL_ENV | {
        "SEARCH_AGENT_LLM_MODEL": "unit-model",
        "SEARCH_AGENT_LLM_BASE_URL": "https://unit.example.com/v1",
        "SEARCH_AGENT_LLM_API_KEY": "unit-key",
    }
    config = load_llm_config("search_agent", env)
    assert config.model == "unit-model"
    assert config.base_url == "https://unit.example.com/v1"
    assert config.api_key == "unit-key"


def test_逐字段混合回落():
    env = GLOBAL_ENV | {"SEARCH_AGENT_LLM_MODEL": "unit-model"}
    config = load_llm_config("search_agent", env)
    assert config.model == "unit-model"
    assert config.base_url == "https://global.example.com/v1"
    assert config.api_key == "global-key"


def test_空字符串视为未配置():
    env = GLOBAL_ENV | {"SEARCH_AGENT_LLM_MODEL": "  "}
    config = load_llm_config("search_agent", env)
    assert config.model == "global-model"


def test_全局也缺失时报错并指明变量名():
    env = {"LLM_MODEL": "m", "LLM_BASE_URL": "https://x/v1"}
    with pytest.raises(ValueError, match="CITATION_VALIDATOR_LLM_API_KEY"):
        load_llm_config("citation_validator", env)
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        load_llm_config("citation_validator", env)


def test_base_url剥掉多余的路径后缀():
    env = GLOBAL_ENV | {
        "LLM_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    }
    config = load_llm_config("writing_orchestrator", env)
    assert config.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_base_url剥掉尾部斜杠():
    env = GLOBAL_ENV | {"LLM_BASE_URL": "https://global.example.com/v1/"}
    config = load_llm_config("writing_orchestrator", env)
    assert config.base_url == "https://global.example.com/v1"


def test_非法运行单元名报错():
    with pytest.raises(ValueError, match="未知运行单元"):
        load_llm_config("not_a_unit", GLOBAL_ENV)
