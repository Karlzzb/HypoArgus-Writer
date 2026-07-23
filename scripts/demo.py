#!/usr/bin/env python
"""空转全流程演示脚本：起真实服务，驱动一遍完整写作闭环并渲染书目。

流程：创建任务 → 并发消费业务与 graph_event 双 SSE 流 → 提交混合两类
分支（纯改写 + 补充佐证）的修订意见 → 引文门禁 → 定稿 → 按两种书目
格式渲染最终交付。

缺省为空转模式：确定性假 LLM + 内存存档器 + 打桩检索子智能体
（写作走 rewriter_loop 真实现链路，仅最底层模型调用是假的），
不依赖任何外部设施，可离线复现。
加 --real 切换生产同构模式：真实 LLM 配置（.env 各单元变量）+
Postgres 存档器（HYPOARGUS_PG_DSN）+ Langfuse 上报（LANGFUSE_* 已配置时）。

每次运行额外产出一份构建过程档案（Markdown）落盘，供人工审核：
完整事件流、每章中间产物、逐章 state 演进快照、修订与终审往返、
最终整篇文章与统一重编号书目。缺省写入 var/demo_archive/<thread_id>.md，
可用 --archive PATH 覆盖。
运行到定稿时另落一份成品文档（仅重编号正文 + 参考文献，不含过程记录），
路径为过程档案同名加 -article 后缀。

用法：
    python scripts/demo.py                    # 空转演示
    python scripts/demo.py --real             # 生产同构演示（需 .env 就绪）
    python scripts/demo.py --archive out.md   # 指定档案落盘路径
"""

import argparse
import asyncio
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import httpx  # noqa: E402
import uvicorn  # noqa: E402

# 整体超时：生产同构模式下 framework 阶段本身就要跑数分钟，给足余量。
TIMEOUT = 7200.0

MIXED_FEEDBACK = "引言口吻克制些；第二章补充行业数据佐证"

# 档案缺省落盘目录（相对仓库根，已在 .gitignore 中忽略）。
ARCHIVE_DIR = REPO_ROOT / "var" / "demo_archive"

# 回归基准输入（issue #19 固化）：创建任务的缺省输入来源，保证复跑一致。
# 人培（汇报）基准保留原路径零改动；其他文种的验收基准按文种分目录固化在
# scripts/baselines/<文种>/ 下（issue #28），经 --task 指定驱动对应文种的真实 E2E。
BASELINE_TASK_PATH = REPO_ROOT / "scripts" / "demo_task.baseline.json"


def load_baseline_task(path: Path = BASELINE_TASK_PATH) -> dict[str, str]:
    """读取基准输入的任务载荷（user_intent / user_identity / session_id）。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["task"]

# 防御性脱敏：模型配置摘要中含这些子串的键一律不写入档案。
_SENSITIVE_KEY_MARKERS = ("key", "secret", "token", "password")


def _timing_suffix() -> str:
    """LLM 计时日志开启时给事件行附加时间戳，方便与调用计时对齐。"""
    from llm.llm_json import timing_enabled

    return f" t={time.time():.1f}" if timing_enabled() else ""


def _fmt_ts(epoch: float) -> str:
    """本地时间的可读格式，档案元信息与业务事件行共用。"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch))


def _short(event_id: str | None) -> str:
    """事件 id 截短为 8 位展示；空值渲染为占位符。"""
    return event_id[:8] if event_id else "-"


class ArchiveRecorder:
    """构建过程档案收集器：贯穿一次演示运行，结束时整体渲染为 Markdown 落盘。

    只做旁路记录，不改变任何驱动逻辑；正文全文一律来自 REST 读
    （finalized 载荷与书目渲染接口），事件信封本身不含正文。
    """

    def __init__(self, real: bool, path_override: str | None) -> None:
        self.real = real
        self.path_override = path_override
        self.thread_id: str | None = None
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.graph_events: list[dict[str, Any]] = []
        self.business_events: list[dict[str, Any]] = []
        self.review_actions: list[dict[str, Any]] = []
        # 每次人工中断点时经书目接口抓取的整篇渲染快照（统一重编号后文本）。
        self.round_snapshots: list[dict[str, Any]] = []
        self.finalized: dict[str, Any] | None = None
        self.bibliographies: dict[str, dict[str, Any]] = {}

    # ---- 采集入口 ----

    def record_graph_event(self, envelope: dict[str, Any]) -> None:
        self.graph_events.append(envelope)

    def record_business_event(self, event: dict[str, Any]) -> None:
        self.business_events.append({"at": time.time(), **event})

    def record_review_action(self, action: dict[str, Any]) -> None:
        self.review_actions.append({"at": time.time(), **action})

    def record_round_snapshot(
        self, round_no: int, rendered: dict[str, Any]
    ) -> None:
        self.round_snapshots.append(
            {"round": round_no, "at": time.time(), "rendered": rendered}
        )

    # ---- 落盘 ----

    def resolve_path(self) -> Path:
        if self.path_override:
            return Path(self.path_override)
        name = self.thread_id or time.strftime("%Y%m%d-%H%M%S")
        return ARCHIVE_DIR / f"{name}.md"

    def write(self) -> Path:
        path = self.resolve_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._render(), encoding="utf-8")
        return path

    def write_article(self) -> Path | None:
        """成品文档单独落盘：仅重编号正文与书目，不含过程记录。

        未到定稿（无章节可渲染）时不落盘，返回 None。
        路径为过程档案同目录同名加 -article 后缀。
        """
        rendered = self.bibliographies.get("gbt7714") or {}
        chapters = rendered.get("chapters") or (self.finalized or {}).get(
            "chapters", []
        )
        if not chapters:
            return None
        archive_path = self.resolve_path()
        path = archive_path.with_name(f"{archive_path.stem}-article.md")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            self._article_text(chapters, rendered.get("bibliography") or []),
            encoding="utf-8",
        )
        return path

    def write_round_article(
        self, round_no: int, rendered: dict[str, Any]
    ) -> Path | None:
        """人工反馈前/各中断点的初稿快照单独落盘，供人在反馈前审阅当版全文。

        路径为过程档案同名加 -article-rN 后缀；第 1 次中断点即人工反馈前
        的那一版原文。无章节可渲染时不落盘，返回 None。
        """
        chapters = (rendered or {}).get("chapters") or []
        if not chapters:
            return None
        archive_path = self.resolve_path()
        path = archive_path.with_name(
            f"{archive_path.stem}-article-r{round_no}.md"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            self._article_text(chapters, rendered.get("bibliography") or []),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _article_text(
        chapters: list[dict[str, Any]], bibliography: list[dict[str, Any]]
    ) -> str:
        """正文 + 参考文献的纯文本拼装，成品与各轮初稿快照共用。"""
        lines: list[str] = []
        for chapter in chapters:
            lines += [chapter["text"], ""]
        if bibliography:
            lines += ["## 参考文献", ""]
            lines += [f"{entry['text']}" for entry in bibliography]
            lines.append("")
        return "\n".join(lines)

    # ---- 渲染 ----

    def _render(self) -> str:
        parts = [
            self._render_meta(),
            self._render_event_stream(),
            self._render_chapter_artifacts(),
            self._render_state_evolution(),
            self._render_review_round(),
            self._render_final_deliverables(),
        ]
        return "\n\n".join(parts) + "\n"

    def _render_meta(self) -> str:
        mode = "真实模型（--real）" if self.real else "空转（确定性假 LLM）"
        finished = _fmt_ts(self.finished_at) if self.finished_at else "（未正常结束）"
        lines = [
            "# 构建过程档案",
            "",
            "## 运行元信息",
            "",
            f"- 运行模式：{mode}",
            f"- thread_id：`{self.thread_id or '（任务未创建）'}`",
            f"- 开始时间：{_fmt_ts(self.started_at)}",
            f"- 结束时间：{finished}",
            "- 模型单元配置摘要（来自 llm_config_used 事件，已脱敏，不含任何密钥）：",
        ]
        configs = self._unit_configs()
        if configs:
            for unit, payload in sorted(configs.items()):
                lines.append(f"  - `{unit}`：`{_compact(payload)}`")
        else:
            lines.append("  - （运行期间未观察到 llm_config_used 事件）")
        return "\n".join(lines)

    def _unit_configs(self) -> dict[str, dict[str, Any]]:
        """按单元汇总 llm_config_used 载荷；含敏感字样的键防御性剔除。"""
        configs: dict[str, dict[str, Any]] = {}
        for envelope in self.graph_events:
            if envelope["type"] != "llm_config_used":
                continue
            payload = {
                key: value
                for key, value in envelope["payload"].items()
                if not any(marker in key.lower() for marker in _SENSITIVE_KEY_MARKERS)
            }
            configs[envelope["unit"]] = payload
        return configs

    def _render_event_stream(self) -> str:
        lines = [
            "## 完整事件流",
            "",
            "### graph_event 可视化通道",
            "",
            "逐条按到达顺序记录：事件类型、单元、关键载荷字段与父子链",
            "（id/parent 为 event_id 前 8 位，parent 指向父事件，可据此审计执行拓扑）。",
            "",
        ]
        if not self.graph_events:
            lines.append("（未收到任何 graph_event。）")
        for index, envelope in enumerate(self.graph_events, start=1):
            lines.append(
                f"{index:>3}. `{envelope['ts']}` **{envelope['type']}** "
                f"unit=`{envelope['unit']}` "
                f"payload=`{_compact(envelope['payload'])}` "
                f"id=`{_short(envelope['event_id'])}` "
                f"parent=`{_short(envelope.get('parent_id'))}`"
            )
        lines += ["", "### 业务 SSE 通道", ""]
        if not self.business_events:
            lines.append("（未收到任何业务事件。）")
        for index, event in enumerate(self.business_events, start=1):
            lines.append(
                f"{index:>3}. `{_fmt_ts(event['at'])}` **{event['type']}** "
                f"data=`{_business_data_digest(event)}`"
            )
        return "\n".join(lines)

    def _rewriter_calls(self) -> list[dict[str, Any]]:
        """从事件流重建每次 rewriter_loop 调用：启动信封 + 其下进度步骤。"""
        calls: list[dict[str, Any]] = []
        for envelope in self.graph_events:
            if (
                envelope["type"] == "subagent_start"
                and envelope["unit"] == "rewriter_loop"
            ):
                calls.append({"start": envelope, "steps": []})
            elif envelope["type"] == "progress" and envelope["unit"] == "rewriter_loop":
                for call in calls:
                    if envelope.get("parent_id") == call["start"]["event_id"]:
                        call["steps"].append(envelope)
                        break
        return calls

    def _round_texts(self, round_index: int) -> dict[str, str]:
        """第 N 次人工中断点快照的章节文本（统一重编号后），无快照则为空。"""
        if round_index >= len(self.round_snapshots):
            return {}
        rendered = self.round_snapshots[round_index]["rendered"]
        return {
            chapter["chapter_id"]: chapter["text"]
            for chapter in rendered.get("chapters", [])
        }

    def _render_chapter_artifacts(self) -> str:
        lines = [
            "## 每章中间产物",
            "",
            "说明：事件信封按设计绝不携带正文全文，且服务未暴露逐章 state 的",
            "REST 读接口，故本节由「事件流推导 + 人工中断点时的整篇渲染快照 +",
            "定稿载荷」三路拼合——草稿正文取自首次人工中断点的书目接口渲染",
            "（已统一重编号），self_check 的 citations_ok 与 issues 按各写作调用的",
            "lint_done / audit_done / revise_triggered 进度事件推导（明细文本未经",
            "REST 暴露时以计数呈现），是否触发修订以 revise_triggered 事件为准；",
            "触发修订时末次 lint_done / audit_done 为修后复检计数（ADR-0004），",
            "citations_ok 按末次（终态）计数推导。",
            "",
        ]
        chapters = (self.finalized or {}).get("chapters", [])
        calls = self._rewriter_calls()
        first_round_texts = self._round_texts(0)
        if not chapters:
            lines.append("（运行未到定稿，无法枚举章节。）")
            return "\n".join(lines)
        for chapter in chapters:
            chapter_id = chapter["chapter_id"]
            lines += [f"### 章节 {chapter_id}", ""]
            chapter_calls = [
                call
                for call in calls
                if call["start"]["payload"].get("chapter_id") == chapter_id
            ]
            for number, call in enumerate(chapter_calls, start=1):
                mode = call["start"]["payload"].get("mode")
                digest = self._call_digest(call)
                revised = "是" if digest["revise_triggered"] else "否"
                lines += [
                    f"- 写作调用 {number}（mode=`{mode}`）：",
                    f"  - 步骤流：{digest['flow'] or '（无进度事件）'}",
                    f"  - lint 违规数：{digest['lint_violations']}，"
                    f"自审 issue 数：{digest['audit_issues']}，"
                    f"退化：{digest['degraded']}",
                    f"  - 是否触发修订（恰好一次修订机制）：{revised}",
                    f"  - self_check 推导结论：citations_ok≈"
                    f"{digest['citations_ok']}，issues 计数={digest['issue_total']}",
                ]
            if not chapter_calls:
                lines.append("- （事件流中未观察到该章的 rewriter_loop 调用。）")
            draft_text = first_round_texts.get(chapter_id)
            lines += [
                "- 草稿正文全文（首次人工中断点渲染快照，已统一重编号）：",
                "",
                "```",
                draft_text if draft_text is not None else "（未捕获到首轮渲染快照）",
                "```",
                "",
                f"- 最终 chapter_summary：{chapter.get('summary', '（定稿载荷未含摘要）')}",
                "",
            ]
        return "\n".join(lines)

    @staticmethod
    def _call_digest(call: dict[str, Any]) -> dict[str, Any]:
        """单次 rewriter_loop 调用的进度事件摘要：步骤流与质检计数。"""
        flow_parts: list[str] = []
        lint_violations: int | str = "未知"
        audit_issues: int | str = "未知"
        degraded = False
        revise_triggered = False
        for step_envelope in call["steps"]:
            payload = step_envelope["payload"]
            step = payload.get("step", "?")
            extras = {
                key: payload[key]
                for key in ("call", "attempts", "text_chars", "violations", "issues")
                if key in payload
            }
            flow_parts.append(f"{step}{_compact(extras) if extras else ''}")
            if step == "lint_done":
                lint_violations = payload.get("violations", "未知")
            elif step == "audit_done":
                audit_issues = payload.get("issues", "未知")
            if payload.get("degraded"):
                degraded = True
            if step == "revise_triggered":
                revise_triggered = True
        clean = lint_violations == 0 and audit_issues == 0 and not degraded
        return {
            "flow": " → ".join(flow_parts),
            "lint_violations": lint_violations,
            "audit_issues": audit_issues,
            "degraded": degraded,
            "revise_triggered": revise_triggered,
            # 末次 lint/audit 计数即终态（触发修订时为修后复检计数，ADR-0004），
            # 故触发过修订不再一票否决 citations_ok。
            "citations_ok": clean,
            "issue_total": (
                (lint_violations if isinstance(lint_violations, int) else 0)
                + (audit_issues if isinstance(audit_issues, int) else 0)
            ),
        }

    def _render_state_evolution(self) -> str:
        lines = [
            "## 逐章 state 演进快照",
            "",
            "快照来源：graph_event 通道的 state_snapshot 事件（每个超步节点更新后",
            "各发布一条，载荷为纯计数元数据）。「已完成章节」由 rewriter_loop 的",
            "subagent_end 事件顺序累积推导。",
            "",
            "| # | 时间 | 单元 | 已完成章节 | 草稿数/总章数 | 引文库条数 | 迭代轮次 | 状态 |",
            "|---|------|------|------------|---------------|------------|----------|------|",
        ]
        done_chapters: list[str] = []
        rows = 0
        for envelope in self.graph_events:
            if (
                envelope["type"] == "subagent_end"
                and envelope["unit"] == "rewriter_loop"
            ):
                chapter_id = envelope["payload"].get("chapter_id")
                if chapter_id and chapter_id not in done_chapters:
                    done_chapters.append(chapter_id)
            if envelope["type"] != "state_snapshot":
                continue
            payload = envelope["payload"]
            rows += 1
            lines.append(
                f"| {rows} | {envelope['ts']} | {envelope['unit']} "
                f"| {'、'.join(done_chapters) or '—'} "
                f"| {payload.get('chapters_completed', '?')}/"
                f"{payload.get('chapter_total', '?')} "
                f"| {payload.get('material_count', '?')} "
                f"| {payload.get('iteration_round', '?')} "
                f"| {payload.get('status', '?')} |"
            )
        if rows == 0:
            lines.append("| — | — | — | — | — | — | — | 未观察到 state_snapshot 事件 |")
        return "\n".join(lines)

    def _render_review_round(self) -> str:
        lines = ["## 修订与终审", ""]
        review_events = [
            event
            for event in self.business_events
            if event["type"] == "review_required"
        ]
        for index, event in enumerate(review_events, start=1):
            data = event["data"]
            lines += [
                f"- 第 {index} 次人工中断点（`{_fmt_ts(event['at'])}`）：",
                f"  - 中断载荷：`{_compact(data)}`",
            ]
        for action in self.review_actions:
            if action["action"] == "revise":
                lines.append(
                    f"- 提交混合修订意见（`{_fmt_ts(action['at'])}`）："
                    f"「{action['feedback']}」"
                )
            else:
                lines.append(f"- 提交定稿（`{_fmt_ts(action['at'])}`）")
        warnings = (self.finalized or {}).get("citation_warnings", [])
        outcome = (
            f"携未决引文警告交付：{warnings}" if warnings else "无未决引文警告，正常交付"
        )
        lines.append(
            f"- 终审结果：{'已定稿，' + outcome if self.finalized else '未到定稿'}"
        )
        return "\n".join(lines)

    def _render_final_deliverables(self) -> str:
        lines = ["## 最终产物", "", "### 整篇文章（统一重编号后）", ""]
        # 正文优先取书目接口的渲染结果（引文标记已统一重编号为 [1][2]…），
        # 未捕获时回退定稿载荷的原始文本（引文为素材 id 标记）。
        rendered = self.bibliographies.get("gbt7714") or {}
        chapters = rendered.get("chapters") or (self.finalized or {}).get(
            "chapters", []
        )
        if not chapters:
            lines.append("（运行未到定稿。）")
        for chapter in chapters:
            lines += [
                f"#### 章节 {chapter['chapter_id']}",
                "",
                "```",
                chapter["text"],
                "```",
                "",
            ]
        for fmt, title in (("gbt7714", "统一重编号书目（gbt7714）"),
                           ("markdown", "书目（markdown）")):
            lines += [f"### {title}", ""]
            fmt_rendered = self.bibliographies.get(fmt)
            if fmt_rendered is None:
                lines.append("（未捕获该格式的渲染结果。）")
                continue
            for entry in fmt_rendered["bibliography"]:
                lines.append(f"- {entry['text']}")
            lines.append("")
        return "\n".join(lines)


def _compact(data: dict[str, Any]) -> str:
    """载荷紧凑单行 JSON：档案行内展示用，保留中文原样。"""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _business_data_digest(event: dict[str, Any]) -> str:
    """业务事件数据的行内摘要：finalized 载荷含全文，只记章节与规模。"""
    data = event["data"]
    if event["type"] == "finalized":
        digest = {
            "chapters": [
                {"chapter_id": c["chapter_id"], "text_chars": len(c["text"])}
                for c in data.get("chapters", [])
            ],
            "citation_warnings": data.get("citation_warnings", []),
        }
        return _compact(digest)
    return _compact(data)


def _build_app(real: bool):
    """按模式构建应用：空转注入假 LLM 与内存存档器，--real 走生产路径。"""
    from service.app import create_app

    if real:
        # rewriter_loop 已是真实现，终审失败的定向回退重写有实际效果，
        # 重试次数走缺省配置（环境变量 DOCUMENT_REVIEW_MAX_RETRIES，缺省 2）。
        return create_app()

    from langgraph.checkpoint.memory import InMemorySaver

    from graph import checkpoint_serializer
    from llm.llm_client import FakeLLM
    from tests.llm_response_plans import (
        FRAMEWORK_KEYED_RESPONSES,
        TRUNK_RESPONSES,
        WRITER_KEYED_RESPONSES,
    )

    from agents.search_agent import make_stub_search_agent

    # 空转也走 rewriter_loop 真实现链路（真编排 + 真校验器 + 真解析）：
    # 写作与自审调用按 WRITER_KEYED_RESPONSES 键控分派，仅最底层模型调用是假的。
    # 检索注入打桩：真实现会调外部检索通道，空转模式不触网。
    fake = FakeLLM(
        list(TRUNK_RESPONSES),
        keyed_responses={**FRAMEWORK_KEYED_RESPONSES, **WRITER_KEYED_RESPONSES},
    )
    return create_app(
        llm_factory=lambda unit: fake,
        checkpointer=InMemorySaver(serde=checkpoint_serializer()),
        search_agent=make_stub_search_agent,
    )


async def _watch_graph_events(
    client: httpx.AsyncClient, thread_id: str, recorder: ArchiveRecorder
) -> None:
    """持续打印 graph_event 可视化通道的事件信封摘要（由主流程取消收尾）。"""
    async with client.stream(
        "GET", f"/graph_events?thread_id={thread_id}"
    ) as response:
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            envelope = json.loads(line[len("data: ") :])
            recorder.record_graph_event(envelope)
            print(
                f"  [graph_event] {envelope['type']:<16} unit={envelope['unit']}"
                f"{_timing_suffix()}"
            )


async def _consume_business(
    client: httpx.AsyncClient,
    thread_id: str,
    recorder: ArchiveRecorder,
    on_review,
    timeout: float = TIMEOUT,
) -> dict | None:
    """消费业务流：打印事件，遇 review_required 交给回调，返回 finalized 载荷。"""

    async def _consume() -> dict | None:
        async with client.stream(
            "GET", f"/tasks/{thread_id}/stream"
        ) as response:
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[len("data: ") :])
                recorder.record_business_event(event)
                data = event["data"]
                if event["type"] == "status":
                    print(
                        f"[业务] 状态 {data['status']}"
                        f"（节点 {data['node']}，第 {data['iteration_round']} 轮）"
                        f"{_timing_suffix()}"
                    )
                elif event["type"] == "review_required":
                    print(f"[业务] 到达人工中断点：章节 {data['chapter_ids']}")
                    await on_review(data)
                elif event["type"] == "finalized":
                    print("[业务] 已定稿")
                    return data
                elif event["type"] == "error":
                    raise RuntimeError(f"运行失败：{data['message']}")
        return None

    return await asyncio.wait_for(_consume(), timeout)


async def _snapshot_round(
    client: httpx.AsyncClient,
    thread_id: str,
    recorder: ArchiveRecorder,
    round_no: int,
) -> None:
    """人工中断点时经书目接口抓取整篇渲染快照，供档案记录该轮章节全文。

    书目接口读的是当前 State 的章节草稿（统一重编号渲染），是唯一能在
    中间轮次拿到章节全文的 REST 读路径；渲染失败不影响主流程。
    """
    try:
        response = await client.get(
            f"/tasks/{thread_id}/bibliography?format=markdown"
        )
        response.raise_for_status()
        rendered = response.json()
        recorder.record_round_snapshot(round_no, rendered)
        # 人工反馈前/各中断点的初稿独立落盘，供人在反馈前审阅当版全文。
        article = recorder.write_round_article(round_no, rendered)
        if article is not None:
            print(f"  [档案] 第 {round_no} 轮初稿已落盘：{article}")
    except httpx.HTTPError as exc:
        print(f"  [档案] 第 {round_no} 轮渲染快照抓取失败（不影响主流程）：{exc}")


def _print_review_commands(thread_id: str, port: int) -> None:
    """人工模式下到达中断点时，打印可用的 REST 提交指令。

    服务端在 LangGraph interrupt 处阻塞等待 resume，本进程不自动提交；
    人在另一终端 curl 后，续跑产生的事件会经同一业务流回到本进程。
    """
    base = f"http://127.0.0.1:{port}"
    print(f"[人工] 服务阻塞于中断点，请在另一终端自行提交审阅动作（base={base}）：")
    print(
        f'  修订：curl -sX POST {base}/tasks/{thread_id}/review'
        ' -H "Content-Type: application/json"'
        ' -d \'{"action":"revise","feedback":"<你的修订意见>"}\''
    )
    print(
        f'  确认：curl -sX POST {base}/tasks/{thread_id}/review'
        ' -H "Content-Type: application/json"'
        ' -d \'{"action":"confirm"}\'   # 大扇出清单确认'
    )
    print(
        f'  定稿：curl -sX POST {base}/tasks/{thread_id}/review'
        ' -H "Content-Type: application/json"'
        ' -d \'{"action":"finalize"}\'   # 跳过修订直接定稿'
    )
    print("[人工] 提交后服务端自动续跑，本进程将在此等待下一事件。")


async def _drive(
    client: httpx.AsyncClient,
    recorder: ArchiveRecorder,
    task_path: Path,
    auto_review: bool,
    port: int,
) -> None:
    """驱动一遍完整闭环并渲染书目。"""
    response = await client.post("/tasks", json=load_baseline_task(task_path))
    response.raise_for_status()
    thread_id = response.json()["thread_id"]
    recorder.thread_id = thread_id
    print(f"任务已创建：thread_id={thread_id}")

    watcher = asyncio.create_task(_watch_graph_events(client, thread_id, recorder))
    reviewed = False
    review_round = 0

    async def on_review(data: dict) -> None:
        nonlocal reviewed, review_round
        # 大扇出确认（issue #49）：意见触及超过大纲一半的章节时，
        # 系统携解析清单重新中断待确认，回复 confirm 后才执行。
        if "pending_confirmation" in data:
            confirmation = data["pending_confirmation"]
            print(
                f"[业务] 修订清单待确认（受影响章节 {confirmation['affected_chapter_ids']}"
                f"/共 {confirmation['total_chapters']} 章）"
            )
            if auto_review:
                print("[演示] 自动提交确认")
                recorder.record_review_action({"action": "confirm"})
                await client.post(
                    f"/tasks/{thread_id}/review", json={"action": "confirm"}
                )
            else:
                _print_review_commands(thread_id, port)
            return
        review_round += 1
        if data.get("citation_warnings"):
            print(f"[业务] 未决引文警告：{data['citation_warnings']}")
        if data.get("review_warnings"):
            print(f"[业务] 篇级评审提示：{data['review_warnings']}")
        await _snapshot_round(client, thread_id, recorder, review_round)
        if not auto_review:
            _print_review_commands(thread_id, port)
            return
        if not reviewed:
            reviewed = True
            print(f"[演示] 提交混合修订意见：{MIXED_FEEDBACK}")
            recorder.record_review_action(
                {"action": "revise", "feedback": MIXED_FEEDBACK}
            )
            await client.post(
                f"/tasks/{thread_id}/review",
                json={"action": "revise", "feedback": MIXED_FEEDBACK},
            )
        else:
            print("[演示] 提交定稿")
            recorder.record_review_action({"action": "finalize"})
            await client.post(
                f"/tasks/{thread_id}/review", json={"action": "finalize"}
            )

    try:
        finalized = await _consume_business(client, thread_id, recorder, on_review)
        # 给可视化通道留一小段排空时间，让定稿前后的尾部事件进入档案。
        await asyncio.sleep(0.5)
    finally:
        watcher.cancel()

    assert finalized is not None
    recorder.finalized = finalized
    for chapter in finalized["chapters"]:
        print(f"\n===== 章节 {chapter['chapter_id']} =====\n{chapter['text']}")

    for fmt in ("gbt7714", "markdown"):
        response = await client.get(
            f"/tasks/{thread_id}/bibliography?format={fmt}"
        )
        response.raise_for_status()
        rendered = response.json()
        recorder.bibliographies[fmt] = rendered
        print(f"\n===== 书目（{fmt}）=====")
        for entry in rendered["bibliography"]:
            print(entry["text"])


async def _main(
    real: bool,
    archive_path: str | None,
    task_path: Path,
    auto_review: bool,
    bind_port: int,
) -> None:
    recorder = ArchiveRecorder(real, archive_path)
    app = _build_app(real)
    config = uvicorn.Config(
        app, host="127.0.0.1", port=bind_port, log_level="warning",
        timeout_graceful_shutdown=3,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        async with asyncio.timeout(600):
            while not server.started:
                await asyncio.sleep(0.02)
        port = server.servers[0].sockets[0].getsockname()[1]
        mode = "生产同构" if real else "空转"
        gate_mode = "自动模拟人工门" if auto_review else "人工门（需自行 curl）"
        print(f"服务已就绪：http://127.0.0.1:{port}（{mode}模式，{gate_mode}）")
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}",
            # SSE 长连接不设 read timeout：业务阶段可能长时间无事件，
            # 靠 _consume_business 的整体超时兜底。
            timeout=httpx.Timeout(100.0, read=None),
        ) as client:
            await _drive(client, recorder, task_path, auto_review, port)
    finally:
        server.should_exit = True
        thread.join(timeout=100)
        # 无论成败都落盘：异常中止时档案保留已采集部分，便于事后排查。
        recorder.finished_at = time.time()
        archived = recorder.write()
        print(f"\n构建过程档案已落盘：{archived}")
        article = recorder.write_article()
        if article is not None:
            print(f"成品文档已落盘：{article}")
    print("\n演示完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real",
        action="store_true",
        help="生产同构模式：真实 LLM 配置 + Postgres 存档器 + Langfuse 上报",
    )
    parser.add_argument(
        "--archive",
        metavar="PATH",
        default=None,
        help="构建过程档案落盘路径（缺省 var/demo_archive/<thread_id>.md）",
    )
    parser.add_argument(
        "--task",
        metavar="PATH",
        type=Path,
        default=BASELINE_TASK_PATH,
        help="任务基准输入路径（缺省人培汇报基准；其他文种见 scripts/baselines/<文种>/）",
    )
    parser.add_argument(
        "--auto-review",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否自动模拟人工门（缺省开）：到达中断点时脚本自动提交"
        " revise→confirm→finalize；传 --no-auto-review 改为人工模式，"
        "仅打印 REST 指令，由人自行 curl 提交，服务端续跑后本进程继续等待。",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="服务监听端口（缺省 0=随机分配；人工模式建议固定以便 curl）",
    )
    args = parser.parse_args()
    asyncio.run(
        _main(args.real, args.archive, args.task, args.auto_review, args.port)
    )
