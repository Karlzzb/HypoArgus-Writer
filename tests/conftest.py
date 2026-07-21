"""测试会话级兜底：剥离 .env 泄漏的 Langfuse 配置。

llm_config 在导入期执行 load_dotenv，会把开发者本地 .env 的 LANGFUSE_*
变量带进测试进程：全量测试将悄悄启用 Langfuse 上报（指向不可达实例），
并抢先安装全局 OTel TracerProvider，破坏可观测测试的导出器注入。
测试必须确定性离线，故会话开始即剥离；需要启用路径的测试自行注入客户端。
"""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _strip_langfuse_env() -> None:
    for name in (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
        "LANGFUSE_TIMEOUT",
        "LANGFUSE_TRACING_ENABLED",
    ):
        os.environ.pop(name, None)
