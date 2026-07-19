#!/usr/bin/env python
"""单次 LLM 延迟探针：复刻论点假说生成的真实提示词，
打印 usage 明细（含思考 token），供性能调测对比开关思考模式的延迟差异。

用法：uv run python scripts/probe_llm_latency.py [--no-thinking]
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from openai import OpenAI  # noqa: E402

from llm.llm_config import load_llm_config  # noqa: E402
from llm.llm_json import JSON_ONLY_RULE  # noqa: E402
from nodes.framework_orchestrator import _ANGLE_GUIDE  # noqa: E402

SYSTEM = (
    "你是假说生成器。针对章节语境下的单个论点，从六角度发散生成可证伪的假说："
    "假设、失效模式、边界条件、竞争解释、预言、反事实。\n"
    + _ANGLE_GUIDE
    + "\n硬性要求：每条假说必须可证伪并声明具体证伪条件——"
    "没有失效条件的命题是观点而非假说，必须锐化或舍弃；"
    "全组做差异去重，每条与其余各条在主张上不同，而非措辞不同；"
    "逐条自评证据可检索性：公开网络或文献能否检索到支撑或反驳该假说的证据。"
    "数量不超过 3 条。"
    "输出 JSON 数组，逐条一项："
    '{"text": "假说表述", "refute_condition": "证伪条件", '
    '"angle": "六角度之一", "evidence_retrievable": true|false}。'
    + JSON_ONLY_RULE
)
USER = (
    "用户写作需求：按「人才培养方案总结（汇报）模版」，为智能网联汽车技术专业"
    "（460704）2025 级高职专科人才培养方案撰写一份评审汇报用的总结\n"
    "文章品类：人才培养方案总结\n"
    "章节标题：行业定位与核心就业岗位\n"
    "章节子标题：['行业发展趋势', '核心就业岗位', '岗位能力要求']\n\n"
    "待发散的论点：锚定智能网联汽车产业链的细分赛道，"
    "论证专业设置与区域产业升级需求的精准契合。"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="附加 extra_body={'enable_thinking': False} 对比关思考后的延迟",
    )
    args = parser.parse_args()

    config = load_llm_config("framework_orchestrator")
    print(f"model={config.model} base_url={config.base_url}")
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    kwargs: dict[str, Any] = {}
    if args.no_thinking:
        kwargs["extra_body"] = {"enable_thinking": False}

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ],
        **kwargs,
    )
    elapsed = time.perf_counter() - t0

    message = response.choices[0].message
    content = message.content or ""
    reasoning = getattr(message, "reasoning_content", None)
    usage = response.usage
    details = getattr(usage, "completion_tokens_details", None)

    print(f"耗时：{elapsed:.1f}s")
    print(f"可见输出：{len(content)} 字符")
    print(f"reasoning_content：{len(reasoning) if reasoning else 0} 字符")
    print(
        f"usage：prompt={usage.prompt_tokens}"
        f" completion={usage.completion_tokens} total={usage.total_tokens}"
    )
    if details is not None:
        print(f"completion_tokens_details：{details}")
    print(f"输出速度：{usage.completion_tokens / elapsed:.0f} tok/s（按 completion 计）")
    print("\n--- 可见输出前 300 字符 ---")
    print(content[:300])


if __name__ == "__main__":
    main()
