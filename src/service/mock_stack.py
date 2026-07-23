"""mock 栈装配档：把 FakeLLM + 打桩子智能体 + 真 make_rewriter_loop
装进 build_graph，产出一个与真栈共用同一 checkpointer 的编译图。

设计要点：
- FakeLLM 在本模块内构造，不读外部 llm_factory——场景库（
  ``mock_scenarios``）提供确定性应答计划，驱动真分支（多章大纲解析 /
  角标正文 / 篇级终审 warn / 退化重试）。
- search_agent 与 chapter_reviewer 用打桩实现（确定性、零 LLM），
  与真栈装配同形地把 ``hook_dispatcher`` 作为 event_hook 注入，
  使子智能体事件经应用内部分发器路由到当前运行 emitter。
- rewriter_loop 用真实现（``make_rewriter_loop``）：逐字流 / 退化重试 /
  风格 lint / 角标形如真。其构造期读 WRITER_DELTA_FLUSH_CHARS /
  WRITER_DELTA_FLUSH_MS 环境变量（与真栈同口径），故调用方（lifespan）
  须在读 env 的时机构造本图。
- build_graph 会对传入的子智能体再包一层 observability，本模块无需处理。
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from agents.chapter_reviewer import make_stub_chapter_reviewer
from agents.rewriter_loop import make_rewriter_loop
from agents.search_agent import make_stub_search_agent
from assembly.assembler_config import AssemblerConfig
from graph import build_graph
from llm.llm_client import FakeLLM
from service.mock_scenarios import DEFAULT_SCENARIO, MockScenario
from service.task_service import SubagentHookDispatcher


def build_mock_graph(
    checkpointer: BaseCheckpointSaver,
    hook_dispatcher: SubagentHookDispatcher,
    *,
    scenario: MockScenario | None = None,
    document_review_max_retries: int | None = None,
    assembler_config: AssemblerConfig | None = None,
) -> CompiledStateGraph:
    """装配 mock 栈编译图。

    ``scenario`` 缺省取 ``DEFAULT_SCENARIO``。``make_rewriter_loop`` 在本
    调用期读 WRITER_DELTA_FLUSH_CHARS / WRITER_DELTA_FLUSH_MS 环境变量
    （与真栈同口径），故调用方须在读 env 的时机构造本图。
    ``document_review_max_retries`` 与 ``assembler_config`` 透传 build_graph，
    未注入时按环境变量取缺省（与真栈一致）。
    """
    scn = scenario or DEFAULT_SCENARIO
    fake = FakeLLM(list(scn.responses), dict(scn.keyed), chunk_size=scn.chunk_size)
    mock_llm_factory = lambda unit: fake  # noqa: E731 - 与 LLMFactory 签名一致，单元名本场景忽略
    return build_graph(
        llm_factory=mock_llm_factory,
        checkpointer=checkpointer,
        search_agent=make_stub_search_agent(hook_dispatcher),
        rewriter_loop=make_rewriter_loop(mock_llm_factory, hook_dispatcher),
        chapter_reviewer=make_stub_chapter_reviewer(hook_dispatcher),
        document_review_max_retries=document_review_max_retries,
        assembler_config=assembler_config,
    )
