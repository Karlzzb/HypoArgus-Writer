"""style_linter 逐规则测试：每条规则至少一个命中用例与一个通过用例。

另覆盖 CJK 断词归一化、YAML/散文双消费自同一文件、章型模板解析与别名、
文种两层合并（列表并集/标量覆盖）、兑底文种与变体分支。
断言按规则名过滤，避免不相关规则的违规造成串扰。
"""

from pathlib import Path

from agents.rewriter_loop import (
    Fact,
    STYLE_GUIDES_DIR,
    Violation,
    count_prose_words,
    detect_chapter_template,
    lint,
    load_config,
    load_prose,
    normalize_cjk_ws,
    resolve_ideology_chapter,
    resolve_template,
    word_count_prompt_block,
)
from agents.contracts import HypothesisPayload, MaterialPayload


def _rules(violations: list[Violation]) -> set[str]:
    """提取违规规则名集合，供命中/通过断言。"""
    return {violation.rule for violation in violations}


def _material(
    material_id: str = "m-h-1",
    hypothesis_id: str = "h-1",
    excerpt: str = "摘录一",
) -> MaterialPayload:
    """构造素材条目，字段与 contracts.MaterialPayload 对齐。"""
    return {
        "id": material_id,
        "hypothesis_id": hypothesis_id,
        "source": "来源一",
        "excerpt": excerpt,
        "relevance_score": 0.9,
        "verdict": "pass",
    }


def _hypothesis(hypothesis_id: str = "h-1") -> HypothesisPayload:
    """构造假说条目，字段与 contracts.HypothesisPayload 对齐。"""
    return {
        "id": hypothesis_id,
        "text": "示例假说",
        "refute_condition": "若无佐证则证伪",
    }


# ---------- 基础设施：归一化 / 双消费 / 模板解析 ----------


def test_归一化_CJK断词_压掉字间空白并保留中英空格() -> None:
    assert normalize_cjk_ws("职 业面 向") == "职业面向"
    assert normalize_cjk_ws("1+X 证 书") == "1+X 证书"
    assert normalize_cjk_ws("面向 AI 行业") == "面向 AI 行业"


def test_归一化_CJK断词_不吞换行结构() -> None:
    assert normalize_cjk_ws("职 业\n面 向") == "职业\n面向"


def test_双消费_YAML与散文_取自同一随包指南目录() -> None:
    cfg = load_config("人才培养方案")
    prose = load_prose("人才培养方案")
    # YAML 侧：机器可读词表可用（通用层口语词 + 文种层章型模板）。
    assert "我们" in cfg["oral_blacklist"]
    assert "学制学位" in cfg["chapter_templates"]
    # 散文侧：不含 YAML 块、保留人话规则描述。
    assert "ssot-config-begin" not in prose
    assert "风格指南" in prose
    # 两侧同源：默认目录即随包目录，通用层与文种层文件均在。
    assert STYLE_GUIDES_DIR.name == "style_guides"
    assert (STYLE_GUIDES_DIR / "通用公文.md").is_file()
    assert (STYLE_GUIDES_DIR / "人才培养方案.md").is_file()


def test_加载配置_缺少SSoT块_抛出异常(tmp_path: Path) -> None:
    guide = tmp_path / "通用公文.md"
    guide.write_text("# 无配置块的指南\n", encoding="utf-8")
    try:
        load_config("通用公文", style_guides_dir=tmp_path)
    except ValueError as error:
        assert "ssot-config" in str(error)
    else:
        raise AssertionError("缺少 ssot-config 块时应抛 ValueError")


def test_加载配置_通用层文件缺失_抛出异常(tmp_path: Path) -> None:
    try:
        load_config("人才培养方案", style_guides_dir=tmp_path)
    except ValueError as error:
        assert "通用公文" in str(error)
    else:
        raise AssertionError("通用层文件缺失时应抛 ValueError")


def test_显式指南目录_lint使用传入目录的词表(tmp_path: Path) -> None:
    guide = tmp_path / "通用公文.md"
    guide.write_text(
        "# 测试指南\n\n<!-- ssot-config-begin\noral_blacklist:\n  - 测试专用词\nssot-config-end -->\n",
        encoding="utf-8",
    )
    violations = lint("正文含测试专用词。", "通用公文", style_guides_dir=tmp_path)
    assert _rules(violations) == {"oral_blacklist"}


def test_章型解析_中文数字前缀标题_取到章型名() -> None:
    assert detect_chapter_template("## 五、职业面向\n\n正文。") == "职业面向"
    assert detect_chapter_template("### （一）子节\n无二级标题。") is None


def test_模板解析_别名归并_高职学制章命中学制学位模板() -> None:
    cfg = load_config("人才培养方案")
    assert resolve_template(cfg, "基本修业年限") is cfg["chapter_templates"]["学制学位"]
    assert resolve_template(cfg, "未登记章型") is None


def test_归位章解析_标题变体_归并到规范章名() -> None:
    cfg = load_config("人才培养方案")
    assert resolve_ideology_chapter(cfg, "培养目标及规格") == "培养目标与培养规格"
    assert resolve_ideology_chapter(cfg, "职业面向") == "职业面向"


# ---------- 机械规则：口语化 / 表格 / 编号 ----------


def test_口语黑名单_出现我们_命中() -> None:
    violations = lint("## 一、总则\n\n本章我们介绍培养定位。", "人才培养方案", "本科")
    assert "oral_blacklist" in _rules(violations)


def test_口语黑名单_公文语感正文_通过() -> None:
    violations = lint("## 一、总则\n\n本专业培养高素质人才。", "人才培养方案", "本科")
    assert "oral_blacklist" not in _rules(violations)


def test_口语黑名单_正则句式_必要条件句式_命中() -> None:
    violations = lint("## 一、总则\n\n实践能力是就业竞争力的必要条件。", "人才培养方案", "本科")
    assert "oral_blacklist" in _rules(violations)


def test_口语黑名单_正则句式_正向预测句式_命中() -> None:
    violations = lint("## 一、总则\n\n课程成绩正向预测岗位胜任力。", "人才培养方案", "本科")
    assert "oral_blacklist" in _rules(violations)


def test_口语黑名单_正则句式_不误伤意识形态必含长段() -> None:
    text = (
        "## 三、培养目标与培养规格\n\n"
        "本专业坚持立德树人，践行社会主义核心价值观，"
        "培养德智体美劳全面发展、具有工匠精神和报国情怀、"
        "面向现代物流行业的高素质技术技能人才。"
    )
    violations = lint(text, "人才培养方案", "本科")
    assert "oral_blacklist" not in _rules(violations)


def test_表格必含_职业面向章无表_命中() -> None:
    violations = lint("## 五、职业面向\n\n本章说明职业面向。", "人才培养方案", "本科")
    assert "table_missing" in _rules(violations)


def test_表格必含_职业面向章有表_通过() -> None:
    text = "## 五、职业面向\n\n| 对应行业 | 岗位 |\n| --- | --- |\n| 物流 | 调度 |\n"
    violations = lint(text, "人才培养方案", "本科")
    assert "table_missing" not in _rules(violations)


def test_编号合规_阿拉伯数字顿号起手_命中() -> None:
    violations = lint("## 一、总则\n\n1、目标定位。", "人才培养方案", "本科")
    assert "numbering" in _rules(violations)


def test_编号合规_中文数字编号_通过() -> None:
    violations = lint("## 一、总则\n\n（一）目标定位。", "人才培养方案", "本科")
    assert "numbering" not in _rules(violations)


# ---------- 模板词表规则：必含 / 禁用 / 禁子项 / 禁用同义词标题 ----------


def test_术语必含_本科培养规格缺思政_命中() -> None:
    text = "## 七、培养规格\n\n| 素质 | 知识 | 能力 |\n| --- | --- | --- |\n| a | b | c |\n"
    violations = lint(text, "人才培养方案", "本科")
    assert "required_terms" in _rules(violations)


def test_术语必含_本科培养规格四分齐备_通过() -> None:
    text = (
        "## 七、培养规格\n\n| 思政 | 素质 | 知识 | 能力 |\n| --- | --- | --- | --- |\n"
        "| a | b | c | d |\n"
    )
    violations = lint(text, "人才培养方案", "本科")
    assert "required_terms" not in _rules(violations)


def test_禁用措辞_本科学制章混入高职措辞_命中() -> None:
    violations = lint("## 三、学制学位\n\n基本修业年限：四年。", "人才培养方案", "本科")
    assert "forbidden_terms" in _rules(violations)


def test_禁用措辞_本科学制章本科口径_通过() -> None:
    violations = lint("## 三、学制学位\n\n标准学制四年，授予工学学士学位。", "人才培养方案", "本科")
    assert "forbidden_terms" not in _rules(violations)


def test_禁用措辞_别名章标题_高职学制章混入本科措辞仍命中() -> None:
    violations = lint("## 三、基本修业年限\n\n标准学制三年。", "人才培养方案", "高职")
    assert "forbidden_terms" in _rules(violations)


def test_禁独立子项_高职培养规格设思政子项_命中() -> None:
    text = (
        "## 七、培养规格\n\n### （一）思政\n\n内容。\n\n| 素质 | 知识 | 能力 |\n"
        "| --- | --- | --- |\n| a | b | c |\n"
    )
    violations = lint(text, "人才培养方案", "高职")
    assert "forbidden_subsection" in _rules(violations)


def test_禁独立子项_高职合并子项思政素质_通过() -> None:
    text = (
        "## 七、培养规格\n\n### （一）思政素质\n\n内容。\n\n| 素质 | 知识 | 能力 |\n"
        "| --- | --- | --- |\n| a | b | c |\n"
    )
    violations = lint(text, "人才培养方案", "高职")
    assert "forbidden_subsection" not in _rules(violations)


def test_禁用同义词标题_职业领域作大章标题_命中() -> None:
    violations = lint("## 五、职业领域\n\n本章说明职业面向。", "人才培养方案", "本科")
    assert "avoid_title" in _rules(violations)


def test_禁用同义词标题_职业面向权威词_通过() -> None:
    text = "## 五、职业面向\n\n| 对应行业 | 岗位 |\n| --- | --- |\n| 物流 | 调度 |\n"
    violations = lint(text, "人才培养方案", "本科")
    assert "avoid_title" not in _rules(violations)


# ---------- 意识形态规则：三态 + 情感语 + 逐字保底 + tier 分支 ----------


def test_政治理论_归位章缺必含串_命中missing() -> None:
    violations = lint("## 三、培养目标与培养规格\n\n培养高素质人才。", "人才培养方案", "本科")
    assert "political_theory_missing" in _rules(violations)


def test_政治理论_变体章标题含必含串_通过missing() -> None:
    text = "## 三、培养目标及规格\n\n践行社会主义核心价值观，弘扬工匠精神。"
    violations = lint(text, "人才培养方案", "本科")
    assert "political_theory_missing" not in _rules(violations)


def test_政治理论_归位章外注入_命中out_of_place() -> None:
    text = "## 五、职业面向\n\n本章践行社会主义核心价值观。"
    violations = lint(text, "人才培养方案", "本科")
    assert "political_theory_out_of_place" in _rules(violations)


def test_政治理论_tier分支_本科专属串在高职文本不校验() -> None:
    text = "## 五、职业面向\n\n坚定四个自信。"
    rules_本科 = _rules(lint(text, "人才培养方案", "本科"))
    rules_高职 = _rules(lint(text, "人才培养方案", "高职"))
    assert "political_theory_out_of_place" in rules_本科
    assert "political_theory_out_of_place" not in rules_高职


def test_政治理论_领域专属串出现在通用文档_命中wrong_domain() -> None:
    violations = lint("## 三、培养目标与培养规格\n\n建设金融强国。", "人才培养方案", "高职")
    assert "political_theory_wrong_domain" in _rules(violations)


def test_政治理论_领域匹配文档_通过wrong_domain() -> None:
    violations = lint(
        "## 三、培养目标与培养规格\n\n建设金融强国。", "人才培养方案", "高职", domain="金融"
    )
    assert "political_theory_wrong_domain" not in _rules(violations)


def test_情感语_培养目标章无情感语_命中() -> None:
    text = "## 三、培养目标与培养规格\n\n践行社会主义核心价值观。"
    violations = lint(text, "人才培养方案", "本科")
    assert "affective_missing" in _rules(violations)


def test_情感语_略松正则_爱国情怀算在位() -> None:
    text = "## 三、培养目标与培养规格\n\n践行社会主义核心价值观，厚植爱国情怀。"
    violations = lint(text, "人才培养方案", "本科")
    assert "affective_missing" not in _rules(violations)


def test_逐字保底_引用触发词但未逐字出全串_命中partial() -> None:
    violations = lint("## 一、总则\n\n贯彻习近平重要论述。", "人才培养方案", "本科")
    assert "political_theory_partial" in _rules(violations)


def test_逐字保底_断词全串经归一化后视为逐字出现_通过partial() -> None:
    text = "## 一、总则\n\n贯彻习近平新 时代中国特色社会主义思想。"
    violations = lint(text, "人才培养方案", "本科")
    assert "political_theory_partial" not in _rules(violations)


# ---------- 事实依据规则：在位 + 查臆造 ----------


def test_事实在位_依据值缺失于正文_命中() -> None:
    violations = lint(
        "## 一、总则\n\n正文未提课程。",
        "人才培养方案",
        "本科",
        references=[Fact(type="course", value="现代物流管理")],
    )
    assert "reference_missing" in _rules(violations)


def test_事实在位_依据值出现于正文_通过() -> None:
    violations = lint(
        "## 一、总则\n\n开设现代物流管理课程。",
        "人才培养方案",
        "本科",
        references=[Fact(type="course", value="现代物流管理")],
    )
    assert "reference_missing" not in _rules(violations)


def test_事实在位_全角数字排版_归一化后通过() -> None:
    violations = lint(
        "## 一、总则\n\n总学分为７８学分。",
        "人才培养方案",
        "本科",
        references=[Fact(type="credit", value="78学分")],
    )
    assert "reference_missing" not in _rules(violations)


def test_事实在位_带空格数值排版_归一化后通过() -> None:
    violations = lint(
        "## 一、总则\n\n总学分为 78 学分，对应行业代码为 (6 4)。",
        "人才培养方案",
        "本科",
        references=[
            Fact(type="credit", value="78 学分"),
            Fact(type="industry_code", value="(64)"),
        ],
    )
    assert "reference_missing" not in _rules(violations)


def test_事实在位_数值类事实确实缺失_归一化后仍命中() -> None:
    violations = lint(
        "## 一、总则\n\n总学分为７８学分。",
        "人才培养方案",
        "本科",
        references=[Fact(type="credit", value="40学分")],
    )
    assert "reference_missing" in _rules(violations)


def test_查臆造_行业代码无依据_命中() -> None:
    violations = lint(
        "## 一、总则\n\n对应行业(64)。",
        "人才培养方案",
        "本科",
        references=[Fact(type="industry_code", value="65")],
    )
    assert "fabricated_industry_code" in _rules(violations)


def test_查臆造_行业代码有依据_通过() -> None:
    violations = lint(
        "## 一、总则\n\n对应行业(64)。",
        "人才培养方案",
        "本科",
        references=[Fact(type="industry_code", value="64")],
    )
    assert "fabricated_industry_code" not in _rules(violations)


def test_查臆造_学分数值无依据_命中() -> None:
    violations = lint(
        "## 一、总则\n\n共计78学分。",
        "人才培养方案",
        "本科",
        references=[Fact(type="credit", value="40学分")],
    )
    assert "fabricated_credit" in _rules(violations)


def test_查臆造_学分数值有依据_通过() -> None:
    violations = lint(
        "## 一、总则\n\n共计78学分。",
        "人才培养方案",
        "本科",
        references=[Fact(type="credit", value="78学分")],
    )
    assert "fabricated_credit" not in _rules(violations)


def test_查臆造_证书名无依据_命中() -> None:
    violations = lint(
        "## 一、总则\n\n取得物流管理职业技能等级证书。",
        "人才培养方案",
        "本科",
        references=[Fact(type="certificate", value="会计职业技能等级证书")],
    )
    assert "fabricated_certificate" in _rules(violations)


def test_查臆造_证书名有依据_通过() -> None:
    violations = lint(
        "## 一、总则\n\n取得物流管理职业技能等级证书。",
        "人才培养方案",
        "本科",
        references=[Fact(type="certificate", value="物流管理职业技能等级证书")],
    )
    assert "fabricated_certificate" not in _rules(violations)


def test_查臆造_无事实依据参数_整组规则不触发() -> None:
    violations = lint("## 一、总则\n\n对应行业(64)，共计78学分。", "人才培养方案", "本科")
    rules = _rules(violations)
    assert "fabricated_industry_code" not in rules
    assert "fabricated_credit" not in rules
    assert "reference_missing" not in rules


def test_查臆造_量化断言_无角标无依据_命中() -> None:
    violations = lint(
        "## 一、总则\n\n经课程改革，实践课时提升30%。",
        "人才培养方案",
        "本科",
        references=[Fact(type="credit", value="78学分")],
    )
    assert "fabricated_quantitative" in _rules(violations)


def test_查臆造_量化断言_参考依据含该数值_通过() -> None:
    violations = lint(
        "## 一、总则\n\n经课程改革，实践课时提升30%。",
        "人才培养方案",
        "本科",
        references=[Fact(type="other", value="实践课时提升 ３０%")],
    )
    assert "fabricated_quantitative" not in _rules(violations)


def test_查臆造_量化断言_同句素材角标_通过() -> None:
    violations = lint(
        "## 一、总则\n\n经课程改革，实践课时提升30%[m-h-1]。",
        "人才培养方案",
        "本科",
        materials=[_material("m-h-1")],
    )
    assert "fabricated_quantitative" not in _rules(violations)


def test_查臆造_量化断言_角标紧随句末标点_通过() -> None:
    violations = lint(
        "## 一、总则\n\n经课程改革，实践课时提升30%。[m-h-1]",
        "人才培养方案",
        "本科",
        materials=[_material("m-h-1")],
    )
    assert "fabricated_quantitative" not in _rules(violations)


def test_查臆造_量化断言_角标在下一句_命中() -> None:
    violations = lint(
        "## 一、总则\n\n经课程改革，实践课时提升30%。改革成效已获评估认可[m-h-1]。",
        "人才培养方案",
        "本科",
        materials=[_material("m-h-1")],
    )
    assert "fabricated_quantitative" in _rules(violations)


def test_查臆造_量化断言_表内数字_不触发() -> None:
    text = (
        "## 一、总则\n\n| 指标 | 数值 |\n| --- | --- |\n| 实践课时提升30% | 达标 |\n"
    )
    violations = lint(text, "人才培养方案", "本科", references=[Fact(type="credit", value="78学分")])
    assert "fabricated_quantitative" not in _rules(violations)


def test_查臆造_量化断言_未传素材与依据_不触发() -> None:
    violations = lint("## 一、总则\n\n经课程改革，实践课时提升30%。", "人才培养方案", "本科")
    assert "fabricated_quantitative" not in _rules(violations)


def test_查臆造_量化断言_倍数与时长单位_命中() -> None:
    violations = lint(
        "## 一、总则\n\n实训产出增长1.5倍，平均实习周期缩短2周。",
        "人才培养方案",
        "本科",
        references=[Fact(type="credit", value="78学分")],
    )
    matched = [v for v in violations if v.rule == "fabricated_quantitative"]
    assert len(matched) == 2


# ---------- 素材角标规则：杜撰 / 重复 / 悬空 / 照抄未标 ----------


def test_素材角标_池外id_命中unknown_material_marker() -> None:
    violations = lint(
        "## 一、总则\n\n结论有据。[m-x-9]",
        "人才培养方案",
        "本科",
        materials=[_material("m-h-1")],
    )
    assert "unknown_material_marker" in _rules(violations)


def test_素材角标_池内id_通过unknown_material_marker() -> None:
    violations = lint(
        "## 一、总则\n\n结论有据。[m-h-1]",
        "人才培养方案",
        "本科",
        materials=[_material("m-h-1")],
    )
    assert "unknown_material_marker" not in _rules(violations)


def test_素材角标_无素材池_不校验角标() -> None:
    violations = lint("## 一、总则\n\n结论有据。[m-x-9]", "人才培养方案", "本科")
    assert "unknown_material_marker" not in _rules(violations)


def test_素材池_重复id_命中duplicate_material_id() -> None:
    violations = lint(
        "## 一、总则\n\n正文。",
        "人才培养方案",
        "本科",
        materials=[_material("m-h-1"), _material("m-h-1")],
    )
    assert "duplicate_material_id" in _rules(violations)


def test_素材池_id唯一_通过duplicate_material_id() -> None:
    violations = lint(
        "## 一、总则\n\n正文。",
        "人才培养方案",
        "本科",
        materials=[_material("m-h-1"), _material("m-h-2")],
    )
    assert "duplicate_material_id" not in _rules(violations)


def test_素材回链_指向不存在假说_命中dangling_hypothesis_id() -> None:
    violations = lint(
        "## 一、总则\n\n正文。",
        "人才培养方案",
        "本科",
        materials=[_material("m-h-1", hypothesis_id="h-9")],
        hypotheses=[_hypothesis("h-1")],
    )
    assert "dangling_hypothesis_id" in _rules(violations)


def test_素材回链_指向既有假说_通过dangling_hypothesis_id() -> None:
    violations = lint(
        "## 一、总则\n\n正文。",
        "人才培养方案",
        "本科",
        materials=[_material("m-h-1", hypothesis_id="h-1")],
        hypotheses=[_hypothesis("h-1")],
    )
    assert "dangling_hypothesis_id" not in _rules(violations)


def test_素材回链_未传假说列表_不校验() -> None:
    violations = lint(
        "## 一、总则\n\n正文。",
        "人才培养方案",
        "本科",
        materials=[_material("m-h-1", hypothesis_id="h-9")],
    )
    assert "dangling_hypothesis_id" not in _rules(violations)


def test_照抄守卫_摘录出现于正文且未挂角标_命中() -> None:
    excerpt = "数字化转型正在重塑物流行业格局"
    violations = lint(
        "## 一、总则\n\n数字化转型，正在重塑物流行业格局。",
        "人才培养方案",
        "本科",
        materials=[_material("m-h-1", excerpt=excerpt)],
    )
    assert "unmarked_derived_content" in _rules(violations)


def test_照抄守卫_摘录出现且已挂角标_通过() -> None:
    excerpt = "数字化转型正在重塑物流行业格局"
    violations = lint(
        "## 一、总则\n\n数字化转型正在重塑物流行业格局。[m-h-1]",
        "人才培养方案",
        "本科",
        materials=[_material("m-h-1", excerpt=excerpt)],
    )
    assert "unmarked_derived_content" not in _rules(violations)


def test_照抄守卫_摘录过短无特征_不触发() -> None:
    violations = lint(
        "## 一、总则\n\n培养高素质人才。",
        "人才培养方案",
        "本科",
        materials=[_material("m-h-1", excerpt="人才")],
    )
    assert "unmarked_derived_content" not in _rules(violations)


def test_违规模型_默认严重级别为error() -> None:
    violation = Violation(rule="示例", message="示例消息")
    assert violation.severity == "error"


# ---------- 字数统计纯函数：各类排除元素 / 混合文本 / 空文本 ----------


def test_字数统计_纯汉字正文() -> None:
    # "本专业培养高素质应用型人才" = 13 汉字（句号不计）。
    assert count_prose_words("本专业培养高素质应用型人才。") == 13


def test_字数统计_汉字加字母数字词() -> None:
    # "ABC" 计1词，"123" 计1词；汉字按字计。
    assert count_prose_words("学制4年，授予工学学士学位。ABC测试123数字。") == 15 + 3


def test_字数统计_排除markdown表格() -> None:
    text = """\
## 一、职业面向

| 对应行业 | 岗位 |
| --- | --- |
| 物流(59) | 调度 |

本专业面向智能制造领域。"""
    # 计标题 "一职业面向" 5字 + 表外散文 "本专业面向智能制造领域" 11字 = 16字。
    assert count_prose_words(text) == 16


def test_字数统计_排除素材角标() -> None:
    # "本专业培养高素质人才" = 10 汉字（句号、角标均不计）。
    assert count_prose_words("本专业培养高素质人才。[m-h-1][m-h-2]") == 10


def test_字数统计_排除行内与块级公式() -> None:
    text = "总学分 $S = 78$ 学分，平均绩点 $$GPA = \\frac{\\sum credits}{n}$$ 计算。"
    # "总学分 学分平均绩点 计算" = 11 汉字（公式内所有内容均不计，包括变量名）。
    assert count_prose_words(text) == 11


def test_字数统计_排除围栏代码块() -> None:
    text = """\
## 一、示例

正文前。

```python
def example():
    return "code"
```

正文后。"""
    # "一示例" 3字 + "正文前" 3字 + "正文后" 3字 = 9字。
    assert count_prose_words(text) == 9


def test_字数统计_排除参考文献与附录() -> None:
    text = """\
## 五、职业面向

本章正文共二十字。

## 参考文献

[1] 某某文献不计入。

## 附录

附录内容也不计。"""
    # "五职业面向" 5字 + "本章正文共二十字" 8字 = 13字（参考文献之后全不计）。
    assert count_prose_words(text) == 13


def test_字数统计_参考文献先于附录_从参考文献处截断() -> None:
    text = "正文。\n## 参考文献\n文献。\n## 附录\n附录。"
    # "正文" = 2字。
    assert count_prose_words(text) == 2


def test_字数统计_混合场景_多类排除元素() -> None:
    text = """\
## 二、培养目标与培养规格

本专业践行社会主义核心价值观，培养德智体美劳全面发展的高素质人才。[m-1]

| 思政 | 素质 |
| --- | --- |
| 表内容 | 不计 |

授予工学学士学位，标准学制4年。公式 $x = 10$ 不计。

```
代码块不计。
```

## 参考文献

[1] 文献不计。"""
    # 标题 "二培养目标与培养规格" 9字
    # + "本专业践行社会主义核心价值观培养德智体美劳全面发展的高素质人才" 30字
    # + 角标 [m-1] 正确移除
    # + "授予工学学士学位标准学制年公式不计" 16字（公式内 x/10 已删）
    # + "4" 1词
    # = 55 汉字 + 4词 = 59
    assert count_prose_words(text) == 59


def test_字数统计_空文本() -> None:
    assert count_prose_words("") == 0


def test_字数统计_纯空白() -> None:
    assert count_prose_words("   \n\n\t  ") == 0


def test_字数统计_纯标点() -> None:
    assert count_prose_words("。，、；：！？") == 0


# ---------- 字数区间规则：三级区间 / 动态收缩 / 节级同级差异 / 表章豁免 ----------


def test_字数规则_章超上限_命中() -> None:
    # 构造一章超上限（5000）。标题「一、总则」= 3 字计入，故正文 4998 字，合计 5001 > 5000。
    text = "## 一、总则\n\n" + "字" * 4998
    violations = lint(text, "人才培养方案", "本科")
    assert "word_count" in _rules(violations)
    assert any("超出上限" in v.message for v in violations if v.rule == "word_count")


def test_字数规则_章不足下限_命中() -> None:
    # 构造一章不足下限（2000）。标题「一、总则」= 3 字计入，故正文 1996 字，合计 1999 < 2000。
    text = "## 一、总则\n\n" + "字" * 1996
    violations = lint(text, "人才培养方案", "本科")
    assert "word_count" in _rules(violations)
    assert any("不足下限" in v.message for v in violations if v.rule == "word_count")


def test_字数规则_章在区间内_通过() -> None:
    # 标题「一、总则」= 3 字计入，故正文 2997 字，合计 3000 在区间 [2000, 5000] 内。
    text = "## 一、总则\n\n" + "字" * 2997
    violations = lint(text, "人才培养方案", "本科")
    assert "word_count" not in _rules(violations)


def test_字数规则_表章豁免下限_只报上限() -> None:
    # 职业面向章为 table_required；构造 1500 字散文（不足章下限 2000 但在表章合理范围）。
    text = "## 五、职业面向\n\n" + "字" * 1500 + "\n\n| 行业 | 岗位 |\n| --- | --- |\n| x | y |\n"
    violations = lint(text, "人才培养方案", "本科")
    # 表章豁免散文下限 → 不报章下限违规；若超上限仍报（这里未超）。
    assert not any(
        v.rule == "word_count" and "不足下限" in v.message for v in violations
    )


def test_字数规则_节超动态收缩上限_命中() -> None:
    # 构造一章含 2 节，节配置上限 1500、动态收缩后上限 = min(1500, 5000÷2) = 1500；
    # 第一节 1600 字超限。
    text = (
        "## 一、总则\n\n"
        + "### （一）第一节\n\n" + "字" * 1600
        + "\n\n### （二）第二节\n\n" + "字" * 1400
    )
    violations = lint(text, "人才培养方案", "本科")
    rules_wc = [v for v in violations if v.rule == "word_count"]
    assert any("第一节" in v.message and "超出上限" in v.message for v in rules_wc)


def test_字数规则_节不足动态收缩下限_命中() -> None:
    # 构造一章 3000 字含 3 节，节配置下限 600、动态收缩后下限 =
    # max(600×0.7, 2000÷3) = max(420, 666.67) = 666.67；第一节 400 字不足。
    text = (
        "## 一、总则\n\n"
        + "### （一）第一节\n\n" + "字" * 400
        + "\n\n### （二）第二节\n\n" + "字" * 1300
        + "\n\n### （三）第三节\n\n" + "字" * 1300
    )
    violations = lint(text, "人才培养方案", "本科")
    rules_wc = [v for v in violations if v.rule == "word_count"]
    assert any("第一节" in v.message and "不足下限" in v.message for v in rules_wc)


def test_字数规则_节级同级差异超倍数_命中() -> None:
    # 构造一章含 2 节，最长 2000、最短 800 → 2000 > 800 × 2 = 1600 → 失衡。
    text = (
        "## 一、总则\n\n"
        + "### （一）短节\n\n" + "字" * 800
        + "\n\n### （二）长节\n\n" + "字" * 2000
    )
    violations = lint(text, "人才培养方案", "本科")
    rules_wc = [v for v in violations if v.rule == "word_count"]
    assert any("失衡" in v.message for v in rules_wc)


def test_字数规则_节级同级差异未超倍数_通过() -> None:
    # 构造一章含 2 节，最长 1600、最短 800 → 1600 = 800 × 2 → 达标。
    text = (
        "## 一、总则\n\n"
        + "### （一）短节\n\n" + "字" * 800
        + "\n\n### （二）长节\n\n" + "字" * 1600
    )
    violations = lint(text, "人才培养方案", "本科")
    # 不报同级差异违规（可能因章/节上下限仍报其他违规，此处只查同级差异）。
    rules_wc = [v for v in violations if v.rule == "word_count"]
    assert not any("失衡" in v.message for v in rules_wc)


def test_字数规则_表章豁免同级差异_不比对() -> None:
    # 职业面向章为 table_required；构造失衡节级（最长 2000、最短 500）。
    text = (
        "## 五、职业面向\n\n"
        + "### （一）短节\n\n" + "字" * 500
        + "\n\n### （二）长节\n\n" + "字" * 2000
        + "\n\n| 行业 | 岗位 |\n| --- | --- |\n| x | y |\n"
    )
    violations = lint(text, "人才培养方案", "本科")
    rules_wc = [v for v in violations if v.rule == "word_count"]
    # 表章豁免同级差异比对 → 不报失衡。
    assert not any("失衡" in v.message for v in rules_wc)


def test_字数规则_小节超动态收缩上限_命中() -> None:
    # 构造一节含 3 小节，小节配置上限 500、动态收缩后上限 =
    # min(500, 5000÷3) = 500；第一小节 600 字超限。
    text = (
        "## 一、总则\n\n### （一）第一节\n\n"
        + "#### 1. 小节一\n\n" + "字" * 600
        + "\n\n#### 2. 小节二\n\n" + "字" * 400
        + "\n\n#### 3. 小节三\n\n" + "字" * 400
    )
    violations = lint(text, "人才培养方案", "本科")
    rules_wc = [v for v in violations if v.rule == "word_count"]
    assert any("小节一" in v.message and "超出上限" in v.message for v in rules_wc)


def test_字数规则_小节收缩以节区间为父级_区间不退化() -> None:
    # 小节按父级「节配置区间」（600～1500）收缩，而非章区间（2000～5000）：
    # 2 小节 → 上限 min(500, 1500÷2)=500、下限 max(140, 600÷2)=300 → (300, 500)。
    # 若误用章区间会得到 (500, 500) 点区间，450 字小节被误判不足下限，
    # 修一次循环永不收敛（issue #19 真实 E2E 复跑发现）。
    text = (
        "## 一、总则\n\n### （一）第一节\n\n" + "字" * 200
        + "\n\n#### 1. 小节一\n\n" + "字" * 450
        + "\n\n#### 2. 小节二\n\n" + "字" * 460
        + "\n\n### （二）第二节\n\n" + "字" * 1100
    )
    violations = lint(text, "人才培养方案", "本科")
    rules_wc = [v for v in violations if v.rule == "word_count"]
    assert not any("小节" in v.message for v in rules_wc)


def test_字数规则_小节按节级收缩区间上下限_均命中() -> None:
    # 4 小节 → 下限 max(140, 600÷4=150)=150、上限 min(500, 1500÷4=375)=375；
    # 100 字小节不足下限、500 字小节超出上限，均须命中。
    text = (
        "## 一、总则\n\n### （一）第一节\n\n"
        + "#### 1. 小节一\n\n" + "字" * 100
        + "\n\n#### 2. 小节二\n\n" + "字" * 500
        + "\n\n#### 3. 小节三\n\n" + "字" * 300
        + "\n\n#### 4. 小节四\n\n" + "字" * 300
        + "\n\n### （二）第二节\n\n" + "字" * 1050
    )
    violations = lint(text, "人才培养方案", "本科")
    rules_wc = [v for v in violations if v.rule == "word_count"]
    assert any("小节一" in v.message and "不足下限" in v.message for v in rules_wc)
    assert any("小节二" in v.message and "超出上限" in v.message for v in rules_wc)


def test_字数规则_无章标题_不校验() -> None:
    # 正文无 ## 标题 → 不落入标准章结构 → 不校验字数。
    text = "正文无章标题，不校验字数。"
    violations = lint(text, "人才培养方案", "本科")
    assert "word_count" not in _rules(violations)


def test_字数目标块_叙述章型_取中上限提示() -> None:
    block = word_count_prompt_block("一、总则", "人才培养方案")
    assert "2000～5000" in block
    assert "600～1500" in block
    assert "200～500" in block
    assert "中上限" in block
    assert "表型章" not in block


def test_字数目标块_表章_取中下限且不得凑段() -> None:
    block = word_count_prompt_block("五、职业面向", "人才培养方案")
    assert "2000～5000" in block
    assert "表型章" in block
    assert "中下限" in block
    assert "不得在表外堆砌" in block


def test_字数目标块_无字数配置_返回空串(tmp_path: Path) -> None:
    guide = tmp_path / "通用公文.md"
    guide.write_text(
        "# 指南\n\n<!-- ssot-config-begin\noral_blacklist: []\nssot-config-end -->\n",
        encoding="utf-8",
    )
    block = word_count_prompt_block("一、总则", "通用公文", style_guides_dir=tmp_path)
    assert block == ""


# ---------- 文种两层合并：列表并集 / 标量覆盖 / 兑底 / 回落 / 散文拼接 ----------


def _write_two_layer_guides(tmp_path: Path) -> None:
    """写一对最小两层指南（通用公文 + 自定义文种），供合并语义正反例共用。"""
    (tmp_path / "通用公文.md").write_text(
        "# 通用指南散文\n\n<!-- ssot-config-begin\n"
        "oral_blacklist:\n  - 通用词\n  - 共用词\n"
        "word_count:\n  chapter:\n    min: 100\n    max: 200\n"
        "ssot-config-end -->\n",
        encoding="utf-8",
    )
    (tmp_path / "自定义文种.md").write_text(
        "# 文种指南散文\n\n<!-- ssot-config-begin\n"
        "oral_blacklist:\n  - 文种词\n  - 共用词\n"
        "word_count:\n  chapter:\n    max: 300\n"
        "ssot-config-end -->\n",
        encoding="utf-8",
    )


def test_两层合并_列表并集_通用在前文种追加且去重(tmp_path: Path) -> None:
    _write_two_layer_guides(tmp_path)
    cfg = load_config("自定义文种", style_guides_dir=tmp_path)
    # 并集保序：通用层条目在前，文种层追加未重复条目，重复的「共用词」只留一份。
    assert cfg["oral_blacklist"] == ["通用词", "共用词", "文种词"]


def test_两层合并_嵌套标量覆盖_未覆盖键保留通用值(tmp_path: Path) -> None:
    _write_two_layer_guides(tmp_path)
    cfg = load_config("自定义文种", style_guides_dir=tmp_path)
    # 嵌套映射逐键递归：文种层覆盖 max，未声明的 min 保留通用层值。
    assert cfg["word_count"]["chapter"] == {"min": 100, "max": 300}


def test_两层合并_兑底文种_不受文种层影响(tmp_path: Path) -> None:
    _write_two_layer_guides(tmp_path)
    cfg = load_config("通用公文", style_guides_dir=tmp_path)
    assert cfg["oral_blacklist"] == ["通用词", "共用词"]
    assert cfg["word_count"]["chapter"] == {"min": 100, "max": 200}


def test_兑底文种_基础门禁生效_口语与编号命中() -> None:
    violations = lint("## 一、总则\n\n本章我们介绍定位。\n1、目标。", "通用公文")
    rules = _rules(violations)
    assert "oral_blacklist" in rules
    assert "numbering" in rules


def test_兑底文种_字数门禁生效_章不足下限命中() -> None:
    violations = lint("## 一、总则\n\n正文过短。", "通用公文")
    assert "word_count" in _rules(violations)


def test_兑底文种_文种层规则不生效_学术断言与意识形态不命中() -> None:
    # 学术断言句式正则与意识形态词表按 ADR-0005 落文种层，通用公文不加载。
    text = "## 三、培养目标与培养规格\n\n实践能力是就业竞争力的必要条件。"
    rules = _rules(lint(text, "通用公文"))
    assert "oral_blacklist" not in rules
    assert not any(rule.startswith("political_theory") for rule in rules)
    assert "affective_missing" not in rules


def test_无专属指南文种_汇报材料_回落通用层() -> None:
    assert load_config("汇报材料") == load_config("通用公文")
    assert load_prose("汇报材料") == load_prose("通用公文")


def test_散文加载_两层拼接_通用在前文种在后(tmp_path: Path) -> None:
    _write_two_layer_guides(tmp_path)
    prose = load_prose("自定义文种", style_guides_dir=tmp_path)
    assert prose.index("通用指南散文") < prose.index("文种指南散文")
    assert "ssot-config-begin" not in prose


# ---------- 调研报告文种层：加载合并 / 章型经 State 携带 / 必含禁用 / 情绪词 / 明责任定时限 ----------


def test_调研报告指南_两层合并正确加载_四章型注册且通用层并入() -> None:
    cfg = load_config("调研报告")
    # 四章型全部注册（键为骨架章标题原文，与 docs_templates/调研报告模版.md 一致）。
    for chapter_type in ["监测概述与数据说明", "维度章", "主要发现与问题诊断", "结论与对策建议"]:
        assert chapter_type in cfg["chapter_templates"]
    # 通用层列表并入（口语黑名单）、标量被文种层覆盖（字数区间「待实证」初值）。
    assert "我们" in cfg["oral_blacklist"]
    assert cfg["word_count"]["chapter"] == {"min": 1200, "max": 4000}
    # 散文侧两层拼接可供提示词注入。
    prose = load_prose("调研报告")
    assert "实证分析" in prose
    assert "ssot-config-begin" not in prose


def _survey_lint(text: str, chapter_type: str | None) -> set[str]:
    """按调研报告文种（无变体）跑 lint，返回规则名集合；章型经参数显式携带。"""
    return _rules(lint(text, "调研报告", None, chapter_type=chapter_type))


def test_调研报告_维度章观点标题_经章型携带识别_无表命中() -> None:
    text = "## 二、空间下沉：精准破局「基层人才荒」\n\n数据显示基层就业占比上升。"
    assert "table_missing" in _survey_lint(text, "维度章")


def test_调研报告_维度章观点标题_含表通过() -> None:
    text = (
        "## 二、空间下沉：精准破局「基层人才荒」\n\n数据显示基层就业占比上升。\n\n"
        "| 指标 | 数值 |\n| ---- | ---- |\n| 占比 | 41 |\n"
    )
    assert "table_missing" not in _survey_lint(text, "维度章")


def test_调研报告_维度章_未携带章型_标题反推不识别观点标题() -> None:
    # ADR-0005：维度章观点标题无法靠标题匹配识别——不传章型则章型规则不触发。
    text = "## 二、空间下沉：精准破局「基层人才荒」\n\n数据显示基层就业占比上升。"
    rules = _survey_lint(text, None)
    assert "table_missing" not in rules
    assert "required_terms" not in rules


def test_调研报告_维度章_表章豁免字数下限() -> None:
    # 散文远低于章下限（1200），但维度章标 table_required 享既有表章豁免，仅保上限。
    text = (
        "## 二、空间下沉：精准破局「基层人才荒」\n\n数据显示基层就业占比上升。\n\n"
        "| 指标 | 数值 |\n| ---- | ---- |\n| 占比 | 41 |\n"
    )
    assert "word_count" not in _survey_lint(text, "维度章")


def test_调研报告_监测概述_缺数据来源与样本_必含词命中() -> None:
    text = "## 一、监测概述与数据说明\n\n### （一）监测背景\n\n围绕政策要求开展监测。"
    violations = lint(text, "调研报告", None, chapter_type="监测概述与数据说明")
    messages = [v.message for v in violations if v.rule == "required_terms"]
    assert any("数据来源" in m for m in messages)
    assert any("样本" in m for m in messages)


def test_调研报告_监测概述_必含词齐备通过() -> None:
    text = (
        "## 一、监测概述与数据说明\n\n### （二）数据来源与可靠性\n\n"
        "数据来源为工学云平台，有效样本量覆盖率达标。"
    )
    assert "required_terms" not in _survey_lint(text, "监测概述与数据说明")


def test_调研报告_监测概述_前置对策建议_禁用词命中() -> None:
    text = "## 一、监测概述与数据说明\n\n数据来源与样本齐备，并先给出对策建议若干。"
    assert "forbidden_terms" in _survey_lint(text, "监测概述与数据说明")


def test_调研报告_维度章_混入对策建议_禁用词命中() -> None:
    text = (
        "## 二、空间下沉：精准破局「基层人才荒」\n\n数据显示占比上升，对策建议先行给出。\n\n"
        "| 指标 | 数值 |\n| ---- | ---- |\n| 占比 | 41 |\n"
    )
    assert "forbidden_terms" in _survey_lint(text, "维度章")


def test_调研报告_维度章_缺数据支撑_必含词命中() -> None:
    text = (
        "## 二、空间下沉：精准破局「基层人才荒」\n\n基层就业占比呈上升态势。\n\n"
        "| 指标 | 占比 |\n| ---- | ---- |\n| 县域就业 | 41% |\n"
    )
    violations = lint(text, "调研报告", None, chapter_type="维度章")
    assert any("数据" in v.message for v in violations if v.rule == "required_terms")


def test_调研报告_结论章_缺核心结论与对策建议节_必含词命中() -> None:
    text = "## 四、总结\n\n全篇工作至此收束。"
    violations = lint(text, "调研报告", None, chapter_type="结论与对策建议")
    messages = [v.message for v in violations if v.rule == "required_terms"]
    assert any("核心结论" in m for m in messages)
    assert any("对策建议" in m for m in messages)


def test_调研报告_结论章_必含词齐备通过() -> None:
    text = _conclusion_chapter(_ACCOUNTABLE_ITEM)
    assert "required_terms" not in _survey_lint(text, "结论与对策建议")


def test_调研报告_发现诊断_缺归因_必含词命中() -> None:
    text = "## 三、主要发现与问题诊断\n\n### （一）领先优势总结\n\n优势明显，短板客观呈现。"
    violations = lint(text, "调研报告", None, chapter_type="主要发现与问题诊断")
    messages = [v.message for v in violations if v.rule == "required_terms"]
    assert any("归因" in m for m in messages)
    assert not any("优势" in m for m in messages)


def test_调研报告_情绪词黑名单_令人振奋与深感担忧命中() -> None:
    text = "## 二、维度分析\n\n监测数据令人振奋，个别指标走弱令人深感担忧。"
    violations = lint(text, "调研报告", None)
    hits = [v.message for v in violations if v.rule == "oral_blacklist"]
    assert any("令人振奋" in m for m in hits)
    assert any("深感担忧" in m for m in hits)


def test_调研报告_情绪词黑名单_中性实证表述通过() -> None:
    text = "## 二、维度分析\n\n数据显示毕业去向落实率为百分之九十六，比去年下降二个百分点。"
    assert "oral_blacklist" not in _survey_lint(text, None)


def test_调研报告_情绪词属文种层_通用公文不命中() -> None:
    text = "## 二、维度分析\n\n监测数据令人振奋。"
    assert "oral_blacklist" not in _rules(lint(text, "通用公文"))


def test_调研报告_学术断言句式_文种层各持一份命中() -> None:
    # ADR-0005：学术断言禁令下沉文种层，人培与调研报告各持一份。
    text = "## 二、维度分析\n\n实践能力是就业竞争力的必要条件。"
    assert "oral_blacklist" in _survey_lint(text, None)


_ACCOUNTABLE_ITEM = "1. 深化县域订单式培养，教务处牵头，各二级学院落实，2026年9月前启动。"


def _conclusion_chapter(*items: str) -> str:
    """构造「结论与对策建议」章：核心结论节 + 对策建议节（条目由参数给定）。"""
    body = "\n".join(items)
    return (
        "## 四、结论与对策建议\n\n### （一）核心结论\n\n"
        "学校人才培养在空间维度呈现下沉特征。\n\n### （二）对策建议\n\n" + body + "\n"
    )


def test_调研报告_对策建议_明责任定时限齐备通过() -> None:
    text = _conclusion_chapter(
        _ACCOUNTABLE_ITEM,
        "2. 建设县域实训基地，实训中心负责，一年内完成布点。",
    )
    assert "accountability" not in _survey_lint(text, "结论与对策建议")


def test_调研报告_对策建议_缺牵头责任命中() -> None:
    text = _conclusion_chapter("1. 深化县域订单式培养，2026年9月前启动。")
    violations = lint(text, "调研报告", None, chapter_type="结论与对策建议")
    messages = [v.message for v in violations if v.rule == "accountability"]
    assert len(messages) == 1
    assert "牵头责任" in messages[0]
    assert "完成时限" not in messages[0]


def test_调研报告_对策建议_缺完成时限命中() -> None:
    text = _conclusion_chapter("1. 深化县域订单式培养，教务处牵头推进。")
    violations = lint(text, "调研报告", None, chapter_type="结论与对策建议")
    messages = [v.message for v in violations if v.rule == "accountability"]
    assert len(messages) == 1
    assert "完成时限" in messages[0]


def test_调研报告_对策建议_裸年份数据引用不算时限() -> None:
    # 「较2024年下降」是数据引用不是完成时限：须 年+月 / 年+底前内 / 期限式。
    text = _conclusion_chapter("1. 优化培养结构，教务处牵头，覆盖面较2024年提升五个百分点。")
    violations = lint(text, "调研报告", None, chapter_type="结论与对策建议")
    messages = [v.message for v in violations if v.rule == "accountability"]
    assert len(messages) == 1
    assert "完成时限" in messages[0]


def test_调研报告_对策建议_泛化落实表述不算牵头责任() -> None:
    # 「落实立德树人」是泛化表述不是责任主体：须「牵头/负责/责任部门」或「XX单位落实」。
    text = _conclusion_chapter("1. 落实立德树人要求，深化订单式培养，2026年9月前启动。")
    violations = lint(text, "调研报告", None, chapter_type="结论与对策建议")
    messages = [v.message for v in violations if v.rule == "accountability"]
    assert len(messages) == 1
    assert "牵头责任" in messages[0]


def test_调研报告_对策建议_逐条判定_仅缺项条目计违规() -> None:
    text = _conclusion_chapter(_ACCOUNTABLE_ITEM, "2. 扩大校企合作范围。")
    violations = lint(text, "调研报告", None, chapter_type="结论与对策建议")
    messages = [v.message for v in violations if v.rule == "accountability"]
    assert len(messages) == 1
    assert "扩大校企合作范围" in messages[0]


def test_调研报告_对策建议_四级子项形式同样判定() -> None:
    text = (
        "## 四、结论与对策建议\n\n### （一）核心结论\n\n结论一句话。\n\n"
        "### （二）对策建议\n\n#### 1. 深化订单式培养\n\n教务处牵头，2026年9月前启动。\n\n"
        "#### 2. 建设实训基地\n\n扩大布点范围。\n"
    )
    violations = lint(text, "调研报告", None, chapter_type="结论与对策建议")
    messages = [v.message for v in violations if v.rule == "accountability"]
    assert len(messages) == 1
    assert "建设实训基地" in messages[0]


def test_调研报告_对策建议节缺失_明责任定时限规则报缺() -> None:
    # 节被裁（首尾章被 LLM 裁剪的留白场景）：章标题含「对策建议」子串使必含词兜不住，
    # 由 accountability 规则直接报节级缺失。
    text = "## 四、结论与对策建议\n\n### （一）核心结论\n\n结论一句话。"
    violations = lint(text, "调研报告", None, chapter_type="结论与对策建议")
    messages = [v.message for v in violations if v.rule == "accountability"]
    assert len(messages) == 1
    assert "未检出" in messages[0]


def test_调研报告_字数目标块_维度章经章型取表型章口径() -> None:
    block = word_count_prompt_block(
        "二、空间下沉：精准破局「基层人才荒」", "调研报告", chapter_type="维度章"
    )
    assert "1200～4000" in block
    assert "表型章" in block


# ---------- 调研报告：数据断言强制角标（第二层，references 数值匹配不豁免） ----------


def test_调研报告_量化断言_references含数值_不豁免_仍命中() -> None:
    # 与人培的关键差异：references 数值有据也不放行，正文可见溯源必须同句角标。
    violations = lint(
        "## 二、空间下沉：精准破局「基层人才荒」\n\n基层就业占比提升30%。",
        "调研报告",
        None,
        chapter_type="维度章",
        references=[Fact(type="other", value="基层就业占比提升 ３０%")],
    )
    assert "fabricated_quantitative" in _rules(violations)


def test_调研报告_量化断言_同句素材角标_通过() -> None:
    violations = lint(
        "## 二、空间下沉：精准破局「基层人才荒」\n\n基层就业占比提升30%[m-h-1]。",
        "调研报告",
        None,
        chapter_type="维度章",
        materials=[_material("m-h-1")],
    )
    assert "fabricated_quantitative" not in _rules(violations)


def test_调研报告_量化断言_表内数字_不触发() -> None:
    # 表内数字由表承载的既有语义在严格分支下保持不变。
    text = (
        "## 二、空间下沉：精准破局「基层人才荒」\n\n"
        "| 指标 | 数值 |\n| ---- | ---- |\n| 基层就业占比提升30% | 达标 |\n"
    )
    violations = lint(
        text,
        "调研报告",
        None,
        chapter_type="维度章",
        references=[Fact(type="other", value="其他依据")],
    )
    assert "fabricated_quantitative" not in _rules(violations)


def test_调研报告_量化断言严格开关_经文种覆盖语义生效() -> None:
    # 开关是标量：通用层二选一（默认豁免），调研报告文种层覆盖为不豁免。
    assert load_config("通用公文")["fabrication"]["quantitative"]["references_exempt"] is True
    assert load_config("调研报告")["fabrication"]["quantitative"]["references_exempt"] is False
    assert load_config("人才培养方案")["fabrication"]["quantitative"]["references_exempt"] is True


def test_调研报告_无变体_人培变体分键词表不受列表形式影响() -> None:
    # 回归护栏：人培 dict[变体→词表] 形式在列表形式支持加入后语义不变。
    text = "## 五、学制学位\n\n标准学制四年，授予工学学士学位。"
    assert "required_terms" not in _rules(lint(text, "人才培养方案", "本科"))
    assert "forbidden_terms" in _rules(
        lint("## 五、学制学位\n\n基本修业年限三年。", "人才培养方案", "本科")
    )
