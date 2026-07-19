"""framework_orchestrator 主节点：论证框架生成的真实业务逻辑。

LLM 调用序列（每步只输出 JSON）：
1. 品类识别与模板匹配（置信度低转自由结构，不报错）；
2. 大纲生成：模板路径只做标题实例化与不适用章节裁剪，自由结构路径直接生成；
3. 全文论点生成（所有章节合并为一次调用）；
4. 逐论点假说生成（每论点一次调用；六角度发散、可证伪、按证据可检索性筛选），
   各章节并发执行（并发度由配置控制），章内论点保持顺序。

全文假说总数配额在发起假说调用前按论点顺序预先分配（每论点
cap = min(单论点上限, 剩余配额)），分配为零的论点不发起 LLM 调用；
与旧串行语义的差异：某论点实际产出少于配额时，差额不再回补给后续论点。

章节 / 论点 / 假说的 ID 与各层数量上限全部由程序保证，不依赖 LLM 自觉；
标题中残留的 {变量} 一律由程序替换为 【待补充：变量名】 占位标记。
LLM 应答解析失败直接抛 ValueError（错误处理与重试是后续 issue）。
"""

import contextvars
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from assembly.assembler_config import AssemblerConfig
from assembly.context_assembler import assemble
from domain.framework_config import FrameworkLimits, load_framework_limits
from llm.llm_client import LLM, LLMFactory
from llm.llm_json import JSON_ONLY_RULE
from llm.llm_json import invoke_json
from domain.state import (
    ArgumentPoint,
    ChapterSpec,
    Hypothesis,
    WorkflowStatus,
    WritingAgentState,
)
from domain.template_skeleton import (
    VARIABLE_PATTERN,
    TemplateSkeleton,
    parse_template_skeleton,
)

# 假说六角度的合法取值；角度不在其中的应答项被程序丢弃。
HYPOTHESIS_ANGLES: tuple[str, ...] = (
    "假设",
    "失效模式",
    "边界条件",
    "竞争解释",
    "预言",
    "反事实",
)

_ANGLE_GUIDE = (
    "六角度含义：\n"
    "- 假设：论点成立必须预先为真的前提信念，每条都是可能为假的承重信念；\n"
    "- 失效模式：论点在实践中可能出错或适得其反的具体方式；\n"
    "- 边界条件：论点在何地、何时、对谁成立与失效，偏向边缘情形；\n"
    "- 竞争解释：能解释同样现象的对立主张，论点存在竞争者；\n"
    "- 预言：论点为真时必然出现的可观察、可核查的后果；\n"
    "- 反事实：若论点为假，世界会呈现的样子。"
)


class FrameworkOrchestratorNode(Protocol):
    """节点函数类型：入参与返回均为图状态（state 具名，满足 LangGraph 节点协议）。"""

    def __call__(self, state: WritingAgentState) -> WritingAgentState: ...


@dataclass(frozen=True)
class _OutlineChapter:
    """大纲阶段的中间产物：实例化后的章节标题与三级子标题文本。"""

    title: str
    subsections: tuple[str, ...]


def _default_templates_dir() -> Path:
    """以本文件位置定位仓库根下的模板目录，不依赖进程工作目录。"""
    return Path(__file__).resolve().parent.parent.parent / "docs_templates"


def _fill_placeholders(text: str) -> str:
    """把残留的 {变量} 替换为醒目占位标记，不阻塞流程。"""
    return VARIABLE_PATTERN.sub(lambda match: f"【待补充：{match.group(1)}】", text)


def _list_template_files(templates_dir: Path) -> list[str]:
    """模板目录下的候选模板文件名（索引文件自身除外）。"""
    return sorted(
        path.name for path in templates_dir.glob("*.md") if path.name != "index.md"
    )


def _identify_genre(
    llm: LLM, templates_dir: Path, user_intent: str, user_identity: str
) -> tuple[str, str | None]:
    """步骤 1 品类识别：返回（品类, 模板文件名或 None）。

    LLM 给出的文件名不在模板目录中时按 None（自由结构）处理，不报错。
    """
    index_path = templates_dir / "index.md"
    index_text = index_path.read_text(encoding="utf-8") if index_path.exists() else ""
    template_files = _list_template_files(templates_dir)
    system = (
        "你是文章品类识别器。根据模板索引、模板文件名列表与用户写作需求，"
        "判断文章品类并匹配最合适的模板。"
        '输出 JSON 对象：{"genre": "品类名", "template_file": "模板文件名或 null"}。'
        "仅当需求与某模板的适用场景明确匹配时才返回该文件名；"
        "把握不大时 template_file 返回 null 转自由结构模式，不要勉强匹配。"
        + JSON_ONLY_RULE
    )
    user = (
        f"模板索引：\n{index_text}\n\n"
        f"模板文件名列表：{template_files}\n\n"
        f"用户身份：{user_identity}\n用户写作需求：{user_intent}"
    )
    payload = invoke_json(llm, "品类识别", system, user, dict)

    genre_raw = payload.get("genre")
    genre = genre_raw.strip() if isinstance(genre_raw, str) else ""
    template_file = payload.get("template_file")
    if not isinstance(template_file, str) or template_file not in template_files:
        return genre, None
    return genre, template_file


def _instantiate_template_outline(
    llm: LLM, skeleton: TemplateSkeleton, user_intent: str, user_identity: str
) -> list[_OutlineChapter]:
    """步骤 2a 模板路径大纲：LLM 只做标题实例化与不适用章节裁剪。

    程序侧按骨架序号对齐强制结构：超界 / 缺失项回落骨架原标题与子标题；
    applicable=false 的章节剔除；残留 {变量} 统一替换为占位标记。
    """
    skeleton_payload = [
        {
            "index": index,
            "numbering": chapter.numbering,
            "title": chapter.title,
            "subsections": [sub.title for sub in chapter.subsections],
        }
        for index, chapter in enumerate(skeleton.chapters, start=1)
    ]
    system = (
        "你是大纲实例化器。给定模板章节骨架与用户写作需求，逐章实例化标题文本。"
        "严格约束：不得增删章节、不得改变骨架结构，只能实例化标题与子标题文字，"
        "以及把明显不适用于本需求的章节标记为 applicable=false 予以裁剪。"
        "模板中形如 {变量名} 的填充变量，用户需求提供了对应信息就代入；"
        "未提供时在标题中原样写 【待补充：变量名】 醒目占位继续，不要追问。"
        "输出 JSON 数组，逐章一项："
        '{"index": 骨架序号, "applicable": true|false, '
        '"title": "实例化标题", "subsections": ["实例化子标题", ...]}。'
        + JSON_ONLY_RULE
    )
    user = (
        f"模板文档标题：{skeleton.doc_title}\n"
        f"模板填充变量：{list(skeleton.variables)}\n"
        f"章节骨架：\n{json.dumps(skeleton_payload, ensure_ascii=False, indent=2)}\n\n"
        f"用户身份：{user_identity}\n用户写作需求：{user_intent}"
    )
    payload = invoke_json(llm, "模板大纲实例化", system, user, list)

    by_index: dict[int, dict[str, Any]] = {}
    for item in payload:
        if isinstance(item, dict) and isinstance(item.get("index"), int):
            by_index.setdefault(item["index"], item)

    chapters: list[_OutlineChapter] = []
    for index, chapter in enumerate(skeleton.chapters, start=1):
        item = by_index.get(index, {})
        if item.get("applicable", True) is False:
            continue
        title_raw = item.get("title")
        if isinstance(title_raw, str) and title_raw.strip():
            title = title_raw.strip()
        else:
            title = chapter.title
        subsections_raw = item.get("subsections")
        if isinstance(subsections_raw, list) and all(
            isinstance(sub, str) for sub in subsections_raw
        ):
            subsections = [sub.strip() for sub in subsections_raw if sub.strip()]
        else:
            subsections = [sub.title for sub in chapter.subsections]
        chapters.append(
            _OutlineChapter(
                title=_fill_placeholders(title),
                subsections=tuple(_fill_placeholders(sub) for sub in subsections),
            )
        )
    return chapters


def _generate_free_outline(
    llm: LLM, user_intent: str, user_identity: str
) -> list[_OutlineChapter]:
    """步骤 2b 自由结构大纲：无模板骨架，LLM 依据需求直接生成章节结构。"""
    system = (
        "你是文章大纲生成器。依据用户写作需求直接生成章节大纲，"
        "章节标题是文章的二级标题，子标题是三级标题。"
        "用户需求未提供的具体信息在标题中写 【待补充：信息名】 醒目占位继续。"
        '输出 JSON 数组，逐章一项：{"title": "章节标题", "subsections": ["子标题", ...]}。'
        + JSON_ONLY_RULE
    )
    user = f"用户身份：{user_identity}\n用户写作需求：{user_intent}"
    payload = invoke_json(llm, "自由结构大纲", system, user, list)

    chapters: list[_OutlineChapter] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        title_raw = item.get("title")
        if not (isinstance(title_raw, str) and title_raw.strip()):
            continue
        subsections_raw = item.get("subsections")
        if isinstance(subsections_raw, list) and all(
            isinstance(sub, str) for sub in subsections_raw
        ):
            subsections = [sub.strip() for sub in subsections_raw if sub.strip()]
        else:
            subsections = []
        chapters.append(
            _OutlineChapter(
                title=_fill_placeholders(title_raw.strip()),
                subsections=tuple(_fill_placeholders(sub) for sub in subsections),
            )
        )
    return chapters


def _generate_points_all(
    llm: LLM,
    user_intent: str,
    user_identity: str,
    genre: str,
    chapters: list[_OutlineChapter],
    max_points: int,
) -> list[list[str]]:
    """步骤 3 全文论点：所有章节合并为一次调用，返回与章节等长的论点文本列表。

    程序侧按章节序号对齐：缺失或非法的章节项按空论点处理，逐章截断至上限。
    """
    chapters_payload = [
        {
            "chapter_index": index,
            "title": chapter.title,
            "subsections": list(chapter.subsections),
        }
        for index, chapter in enumerate(chapters, start=1)
    ]
    system = (
        "你是章节论点生成器。给定全文大纲，一次性为每一章生成中心论点，"
        "每条论点是该章存在的理由之一，是后续假说与检索的锚点；"
        "全文视角下各章论点互不重复、彼此衔接。"
        f"每章数量不超过 {max_points} 条。"
        "输出 JSON 数组，逐章一项："
        '{"chapter_index": 章节序号, "points": [{"text": "论点表述"}, ...]}。'
        + JSON_ONLY_RULE
    )
    user = (
        f"用户身份：{user_identity}\n用户写作需求：{user_intent}\n"
        f"文章品类：{genre or '（未识别）'}\n\n"
        f"全文大纲：\n{json.dumps(chapters_payload, ensure_ascii=False, indent=2)}"
    )
    payload = invoke_json(llm, "全文论点生成", system, user, list)

    by_index: dict[int, list[Any]] = {}
    for item in payload:
        if (
            isinstance(item, dict)
            and isinstance(item.get("chapter_index"), int)
            and isinstance(item.get("points"), list)
        ):
            by_index.setdefault(item["chapter_index"], item["points"])

    points_per_chapter: list[list[str]] = []
    for index in range(1, len(chapters) + 1):
        texts = [
            point["text"].strip()
            for point in by_index.get(index, [])
            if isinstance(point, dict)
            and isinstance(point.get("text"), str)
            and point["text"].strip()
        ]
        points_per_chapter.append(texts[:max_points])
    return points_per_chapter


def _generate_hypotheses(
    llm: LLM,
    user_intent: str,
    genre: str,
    chapter: _OutlineChapter,
    point_text: str,
    max_count: int,
) -> list[tuple[str, str, str]]:
    """步骤 4 逐论点假说：返回（text, refute_condition, angle）三元组列表。

    程序侧筛选：证据不可检索的、角度非法的、字段缺失的一律丢弃，再截断至上限。
    """
    system = (
        "你是假说生成器。针对章节语境下的单个论点，从六角度发散生成可证伪的假说："
        "假设、失效模式、边界条件、竞争解释、预言、反事实。\n"
        + _ANGLE_GUIDE
        + "\n硬性要求：每条假说必须可证伪并声明具体证伪条件——"
        "没有失效条件的命题是观点而非假说，必须锐化或舍弃；"
        "全组做差异去重，每条与其余各条在主张上不同，而非措辞不同；"
        "逐条自评证据可检索性：公开网络或文献能否检索到支撑或反驳该假说的证据。"
        f"数量不超过 {max_count} 条。"
        "输出 JSON 数组，逐条一项："
        '{"text": "假说表述", "refute_condition": "证伪条件", '
        '"angle": "六角度之一", "evidence_retrievable": true|false}。'
        + JSON_ONLY_RULE
    )
    user = (
        f"用户写作需求：{user_intent}\n文章品类：{genre or '（未识别）'}\n"
        f"章节标题：{chapter.title}\n章节子标题：{list(chapter.subsections)}\n\n"
        f"待发散的论点：{point_text}"
    )
    payload = invoke_json(llm, "论点假说生成", system, user, list)

    hypotheses: list[tuple[str, str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        refute_condition = item.get("refute_condition")
        angle = item.get("angle")
        if not (isinstance(text, str) and text.strip()):
            continue
        if not (isinstance(refute_condition, str) and refute_condition.strip()):
            continue
        if angle not in HYPOTHESIS_ANGLES:
            continue
        if item.get("evidence_retrievable") is not True:
            continue
        hypotheses.append((text.strip(), refute_condition.strip(), angle))
    return hypotheses[:max_count]


def _allocate_hypothesis_caps(
    point_counts: list[int], limits: FrameworkLimits
) -> list[list[int]]:
    """按论点顺序预分配全文假说总数配额，返回与各章论点等长的配额列表。

    每论点 cap = min(单论点上限, 剩余配额)，剩余配额按分配额（而非实际产出）扣减，
    使配额在并发发起假说调用前即完全确定。
    """
    remaining_total = limits.max_hypotheses_total
    caps: list[list[int]] = []
    for count in point_counts:
        chapter_caps: list[int] = []
        for _ in range(count):
            cap = min(limits.max_hypotheses_per_point, remaining_total)
            remaining_total -= cap
            chapter_caps.append(cap)
        caps.append(chapter_caps)
    return caps


def make_framework_orchestrator_node(
    llm_factory: LLMFactory,
    templates_dir: Path | None = None,
    limits: FrameworkLimits | None = None,
    assembler_config: AssemblerConfig | None = None,
) -> FrameworkOrchestratorNode:
    """构造 framework_orchestrator 节点函数。

    templates_dir 为 None 时使用仓库根 docs_templates/（以本文件位置定位）；
    limits 为 None 时在节点执行时读取环境变量配置；
    assembler_config 为 None 时在节点执行时读取环境变量装配配置。
    """
    resolved_templates_dir = templates_dir or _default_templates_dir()

    def node(state: WritingAgentState) -> WritingAgentState:
        effective_limits = limits if limits is not None else load_framework_limits()
        llm = llm_factory("framework_orchestrator")
        # 用户意图与身份两段一律经装配入口取得，不再直接读 state。
        context = assemble(
            state, "framework_orchestrator", config=assembler_config
        )
        user_intent = context.text("user_intent")
        user_identity = context.text("user_identity")

        genre, template_file = _identify_genre(
            llm, resolved_templates_dir, user_intent, user_identity
        )
        if template_file is not None:
            skeleton = parse_template_skeleton(
                (resolved_templates_dir / template_file).read_text(encoding="utf-8")
            )
            outline_chapters = _instantiate_template_outline(
                llm, skeleton, user_intent, user_identity
            )
        else:
            outline_chapters = _generate_free_outline(llm, user_intent, user_identity)

        point_texts_per_chapter = _generate_points_all(
            llm,
            user_intent,
            user_identity,
            genre,
            outline_chapters,
            effective_limits.max_points_per_chapter,
        )
        caps_per_chapter = _allocate_hypothesis_caps(
            [len(texts) for texts in point_texts_per_chapter], effective_limits
        )

        def build_chapter(chapter_index: int) -> ChapterSpec:
            """生成单章的论点与假说结构（供并发执行，章内论点保持顺序）。"""
            chapter = outline_chapters[chapter_index]
            chapter_id = f"ch{chapter_index + 1}"
            points: list[ArgumentPoint] = []
            for point_index, point_text in enumerate(
                point_texts_per_chapter[chapter_index]
            ):
                point_id = f"{chapter_id}-p{point_index + 1}"
                cap = caps_per_chapter[chapter_index][point_index]
                hypotheses: list[Hypothesis] = []
                # 预分配配额为零的论点不发起 LLM 调用。
                if cap > 0:
                    raw_hypotheses = _generate_hypotheses(
                        llm, user_intent, genre, chapter, point_text, cap
                    )
                    hypotheses = [
                        Hypothesis(
                            id=f"{point_id}-h{hyp_index + 1}",
                            text=text,
                            refute_condition=refute_condition,
                            angle=angle,
                        )
                        for hyp_index, (text, refute_condition, angle) in enumerate(
                            raw_hypotheses
                        )
                    ]
                points.append(
                    ArgumentPoint(id=point_id, text=point_text, hypotheses=hypotheses)
                )
            return ChapterSpec(
                id=chapter_id,
                title=chapter.title,
                subsections=list(chapter.subsections),
                points=points,
            )

        outline: list[ChapterSpec] = []
        if outline_chapters:
            max_workers = min(
                len(outline_chapters), effective_limits.max_concurrent_chapters
            )
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 每个章节任务复制当前 contextvars 上下文执行，
                # 保证 Langfuse span 父子关系跨工作线程成立。
                futures = [
                    executor.submit(
                        contextvars.copy_context().run, build_chapter, chapter_index
                    )
                    for chapter_index in range(len(outline_chapters))
                ]
                outline = [future.result() for future in futures]

        return WritingAgentState(
            genre=genre,
            template_id=template_file,
            outline=outline,
            status=WorkflowStatus.FRAMEWORK_BUILDING,
            current_node_llm_config={
                "unit": "framework_orchestrator",
                **llm.metadata,
            },
        )

    return node
