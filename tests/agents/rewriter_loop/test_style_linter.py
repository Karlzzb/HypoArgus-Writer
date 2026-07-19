"""style_linter 逐规则测试：每条规则至少一个命中用例与一个通过用例。

另覆盖 CJK 断词归一化、YAML/散文双消费自同一文件、章型模板解析与别名、tier 分支。
断言按规则名过滤，避免不相关规则的违规造成串扰。
"""

from pathlib import Path

from agents.rewriter_loop import (
    DEFAULT_STYLE_GUIDE_PATH,
    Fact,
    Violation,
    detect_chapter_template,
    lint,
    load_config,
    load_prose,
    normalize_cjk_ws,
    resolve_ideology_chapter,
    resolve_template,
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


def test_双消费_YAML与散文_取自同一随包指南文件() -> None:
    cfg = load_config()
    prose = load_prose()
    # YAML 侧：机器可读词表可用。
    assert "我们" in cfg["oral_blacklist"]
    assert "学制学位" in cfg["chapter_templates"]
    # 散文侧：不含 YAML 块、保留人话规则描述。
    assert "ssot-config-begin" not in prose
    assert "风格指南" in prose
    # 两侧同源：默认路径即随包文件。
    assert DEFAULT_STYLE_GUIDE_PATH.name == "style_guide.md"
    assert DEFAULT_STYLE_GUIDE_PATH.exists()


def test_加载配置_缺少SSoT块_抛出异常(tmp_path: Path) -> None:
    guide = tmp_path / "broken.md"
    guide.write_text("# 无配置块的指南\n", encoding="utf-8")
    try:
        load_config(guide)
    except ValueError as error:
        assert "ssot-config" in str(error)
    else:
        raise AssertionError("缺少 ssot-config 块时应抛 ValueError")


def test_显式指南路径_lint使用传入文件的词表(tmp_path: Path) -> None:
    guide = tmp_path / "guide.md"
    guide.write_text(
        "# 测试指南\n\n<!-- ssot-config-begin\noral_blacklist:\n  - 测试专用词\nssot-config-end -->\n",
        encoding="utf-8",
    )
    violations = lint("正文含测试专用词。", "本科", style_guide_path=guide)
    assert _rules(violations) == {"oral_blacklist"}


def test_章型解析_中文数字前缀标题_取到章型名() -> None:
    assert detect_chapter_template("## 五、职业面向\n\n正文。") == "职业面向"
    assert detect_chapter_template("### （一）子节\n无二级标题。") is None


def test_模板解析_别名归并_高职学制章命中学制学位模板() -> None:
    cfg = load_config()
    assert resolve_template(cfg, "基本修业年限") is cfg["chapter_templates"]["学制学位"]
    assert resolve_template(cfg, "未登记章型") is None


def test_归位章解析_标题变体_归并到规范章名() -> None:
    cfg = load_config()
    assert resolve_ideology_chapter(cfg, "培养目标及规格") == "培养目标与培养规格"
    assert resolve_ideology_chapter(cfg, "职业面向") == "职业面向"


# ---------- 机械规则：口语化 / 表格 / 编号 ----------


def test_口语黑名单_出现我们_命中() -> None:
    violations = lint("## 一、总则\n\n本章我们介绍培养定位。", "本科")
    assert "oral_blacklist" in _rules(violations)


def test_口语黑名单_公文语感正文_通过() -> None:
    violations = lint("## 一、总则\n\n本专业培养高素质人才。", "本科")
    assert "oral_blacklist" not in _rules(violations)


def test_表格必含_职业面向章无表_命中() -> None:
    violations = lint("## 五、职业面向\n\n本章说明职业面向。", "本科")
    assert "table_missing" in _rules(violations)


def test_表格必含_职业面向章有表_通过() -> None:
    text = "## 五、职业面向\n\n| 对应行业 | 岗位 |\n| --- | --- |\n| 物流 | 调度 |\n"
    violations = lint(text, "本科")
    assert "table_missing" not in _rules(violations)


def test_编号合规_阿拉伯数字顿号起手_命中() -> None:
    violations = lint("## 一、总则\n\n1、目标定位。", "本科")
    assert "numbering" in _rules(violations)


def test_编号合规_中文数字编号_通过() -> None:
    violations = lint("## 一、总则\n\n（一）目标定位。", "本科")
    assert "numbering" not in _rules(violations)


# ---------- 模板词表规则：必含 / 禁用 / 禁子项 / 禁用同义词标题 ----------


def test_术语必含_本科培养规格缺思政_命中() -> None:
    text = "## 七、培养规格\n\n| 素质 | 知识 | 能力 |\n| --- | --- | --- |\n| a | b | c |\n"
    violations = lint(text, "本科")
    assert "required_terms" in _rules(violations)


def test_术语必含_本科培养规格四分齐备_通过() -> None:
    text = (
        "## 七、培养规格\n\n| 思政 | 素质 | 知识 | 能力 |\n| --- | --- | --- | --- |\n"
        "| a | b | c | d |\n"
    )
    violations = lint(text, "本科")
    assert "required_terms" not in _rules(violations)


def test_禁用措辞_本科学制章混入高职措辞_命中() -> None:
    violations = lint("## 三、学制学位\n\n基本修业年限：四年。", "本科")
    assert "forbidden_terms" in _rules(violations)


def test_禁用措辞_本科学制章本科口径_通过() -> None:
    violations = lint("## 三、学制学位\n\n标准学制四年，授予工学学士学位。", "本科")
    assert "forbidden_terms" not in _rules(violations)


def test_禁用措辞_别名章标题_高职学制章混入本科措辞仍命中() -> None:
    violations = lint("## 三、基本修业年限\n\n标准学制三年。", "高职")
    assert "forbidden_terms" in _rules(violations)


def test_禁独立子项_高职培养规格设思政子项_命中() -> None:
    text = (
        "## 七、培养规格\n\n### （一）思政\n\n内容。\n\n| 素质 | 知识 | 能力 |\n"
        "| --- | --- | --- |\n| a | b | c |\n"
    )
    violations = lint(text, "高职")
    assert "forbidden_subsection" in _rules(violations)


def test_禁独立子项_高职合并子项思政素质_通过() -> None:
    text = (
        "## 七、培养规格\n\n### （一）思政素质\n\n内容。\n\n| 素质 | 知识 | 能力 |\n"
        "| --- | --- | --- |\n| a | b | c |\n"
    )
    violations = lint(text, "高职")
    assert "forbidden_subsection" not in _rules(violations)


def test_禁用同义词标题_职业领域作大章标题_命中() -> None:
    violations = lint("## 五、职业领域\n\n本章说明职业面向。", "本科")
    assert "avoid_title" in _rules(violations)


def test_禁用同义词标题_职业面向权威词_通过() -> None:
    text = "## 五、职业面向\n\n| 对应行业 | 岗位 |\n| --- | --- |\n| 物流 | 调度 |\n"
    violations = lint(text, "本科")
    assert "avoid_title" not in _rules(violations)


# ---------- 意识形态规则：三态 + 情感语 + 逐字保底 + tier 分支 ----------


def test_政治理论_归位章缺必含串_命中missing() -> None:
    violations = lint("## 三、培养目标与培养规格\n\n培养高素质人才。", "本科")
    assert "political_theory_missing" in _rules(violations)


def test_政治理论_变体章标题含必含串_通过missing() -> None:
    text = "## 三、培养目标及规格\n\n践行社会主义核心价值观，弘扬工匠精神。"
    violations = lint(text, "本科")
    assert "political_theory_missing" not in _rules(violations)


def test_政治理论_归位章外注入_命中out_of_place() -> None:
    text = "## 五、职业面向\n\n本章践行社会主义核心价值观。"
    violations = lint(text, "本科")
    assert "political_theory_out_of_place" in _rules(violations)


def test_政治理论_tier分支_本科专属串在高职文本不校验() -> None:
    text = "## 五、职业面向\n\n坚定四个自信。"
    rules_本科 = _rules(lint(text, "本科"))
    rules_高职 = _rules(lint(text, "高职"))
    assert "political_theory_out_of_place" in rules_本科
    assert "political_theory_out_of_place" not in rules_高职


def test_政治理论_领域专属串出现在通用文档_命中wrong_domain() -> None:
    violations = lint("## 三、培养目标与培养规格\n\n建设金融强国。", "高职")
    assert "political_theory_wrong_domain" in _rules(violations)


def test_政治理论_领域匹配文档_通过wrong_domain() -> None:
    violations = lint(
        "## 三、培养目标与培养规格\n\n建设金融强国。", "高职", domain="金融"
    )
    assert "political_theory_wrong_domain" not in _rules(violations)


def test_情感语_培养目标章无情感语_命中() -> None:
    text = "## 三、培养目标与培养规格\n\n践行社会主义核心价值观。"
    violations = lint(text, "本科")
    assert "affective_missing" in _rules(violations)


def test_情感语_略松正则_爱国情怀算在位() -> None:
    text = "## 三、培养目标与培养规格\n\n践行社会主义核心价值观，厚植爱国情怀。"
    violations = lint(text, "本科")
    assert "affective_missing" not in _rules(violations)


def test_逐字保底_引用触发词但未逐字出全串_命中partial() -> None:
    violations = lint("## 一、总则\n\n贯彻习近平重要论述。", "本科")
    assert "political_theory_partial" in _rules(violations)


def test_逐字保底_断词全串经归一化后视为逐字出现_通过partial() -> None:
    text = "## 一、总则\n\n贯彻习近平新 时代中国特色社会主义思想。"
    violations = lint(text, "本科")
    assert "political_theory_partial" not in _rules(violations)


# ---------- 事实依据规则：在位 + 查臆造 ----------


def test_事实在位_依据值缺失于正文_命中() -> None:
    violations = lint(
        "## 一、总则\n\n正文未提课程。",
        "本科",
        references=[Fact(type="course", value="现代物流管理")],
    )
    assert "reference_missing" in _rules(violations)


def test_事实在位_依据值出现于正文_通过() -> None:
    violations = lint(
        "## 一、总则\n\n开设现代物流管理课程。",
        "本科",
        references=[Fact(type="course", value="现代物流管理")],
    )
    assert "reference_missing" not in _rules(violations)


def test_查臆造_行业代码无依据_命中() -> None:
    violations = lint(
        "## 一、总则\n\n对应行业(64)。",
        "本科",
        references=[Fact(type="industry_code", value="65")],
    )
    assert "fabricated_industry_code" in _rules(violations)


def test_查臆造_行业代码有依据_通过() -> None:
    violations = lint(
        "## 一、总则\n\n对应行业(64)。",
        "本科",
        references=[Fact(type="industry_code", value="64")],
    )
    assert "fabricated_industry_code" not in _rules(violations)


def test_查臆造_学分数值无依据_命中() -> None:
    violations = lint(
        "## 一、总则\n\n共计78学分。",
        "本科",
        references=[Fact(type="credit", value="40学分")],
    )
    assert "fabricated_credit" in _rules(violations)


def test_查臆造_学分数值有依据_通过() -> None:
    violations = lint(
        "## 一、总则\n\n共计78学分。",
        "本科",
        references=[Fact(type="credit", value="78学分")],
    )
    assert "fabricated_credit" not in _rules(violations)


def test_查臆造_证书名无依据_命中() -> None:
    violations = lint(
        "## 一、总则\n\n取得物流管理职业技能等级证书。",
        "本科",
        references=[Fact(type="certificate", value="会计职业技能等级证书")],
    )
    assert "fabricated_certificate" in _rules(violations)


def test_查臆造_证书名有依据_通过() -> None:
    violations = lint(
        "## 一、总则\n\n取得物流管理职业技能等级证书。",
        "本科",
        references=[Fact(type="certificate", value="物流管理职业技能等级证书")],
    )
    assert "fabricated_certificate" not in _rules(violations)


def test_查臆造_无事实依据参数_整组规则不触发() -> None:
    violations = lint("## 一、总则\n\n对应行业(64)，共计78学分。", "本科")
    rules = _rules(violations)
    assert "fabricated_industry_code" not in rules
    assert "fabricated_credit" not in rules
    assert "reference_missing" not in rules


# ---------- 素材角标规则：杜撰 / 重复 / 悬空 / 照抄未标 ----------


def test_素材角标_池外id_命中unknown_material_marker() -> None:
    violations = lint(
        "## 一、总则\n\n结论有据。[m-x-9]",
        "本科",
        materials=[_material("m-h-1")],
    )
    assert "unknown_material_marker" in _rules(violations)


def test_素材角标_池内id_通过unknown_material_marker() -> None:
    violations = lint(
        "## 一、总则\n\n结论有据。[m-h-1]",
        "本科",
        materials=[_material("m-h-1")],
    )
    assert "unknown_material_marker" not in _rules(violations)


def test_素材角标_无素材池_不校验角标() -> None:
    violations = lint("## 一、总则\n\n结论有据。[m-x-9]", "本科")
    assert "unknown_material_marker" not in _rules(violations)


def test_素材池_重复id_命中duplicate_material_id() -> None:
    violations = lint(
        "## 一、总则\n\n正文。",
        "本科",
        materials=[_material("m-h-1"), _material("m-h-1")],
    )
    assert "duplicate_material_id" in _rules(violations)


def test_素材池_id唯一_通过duplicate_material_id() -> None:
    violations = lint(
        "## 一、总则\n\n正文。",
        "本科",
        materials=[_material("m-h-1"), _material("m-h-2")],
    )
    assert "duplicate_material_id" not in _rules(violations)


def test_素材回链_指向不存在假说_命中dangling_hypothesis_id() -> None:
    violations = lint(
        "## 一、总则\n\n正文。",
        "本科",
        materials=[_material("m-h-1", hypothesis_id="h-9")],
        hypotheses=[_hypothesis("h-1")],
    )
    assert "dangling_hypothesis_id" in _rules(violations)


def test_素材回链_指向既有假说_通过dangling_hypothesis_id() -> None:
    violations = lint(
        "## 一、总则\n\n正文。",
        "本科",
        materials=[_material("m-h-1", hypothesis_id="h-1")],
        hypotheses=[_hypothesis("h-1")],
    )
    assert "dangling_hypothesis_id" not in _rules(violations)


def test_素材回链_未传假说列表_不校验() -> None:
    violations = lint(
        "## 一、总则\n\n正文。",
        "本科",
        materials=[_material("m-h-1", hypothesis_id="h-9")],
    )
    assert "dangling_hypothesis_id" not in _rules(violations)


def test_照抄守卫_摘录出现于正文且未挂角标_命中() -> None:
    excerpt = "数字化转型正在重塑物流行业格局"
    violations = lint(
        "## 一、总则\n\n数字化转型，正在重塑物流行业格局。",
        "本科",
        materials=[_material("m-h-1", excerpt=excerpt)],
    )
    assert "unmarked_derived_content" in _rules(violations)


def test_照抄守卫_摘录出现且已挂角标_通过() -> None:
    excerpt = "数字化转型正在重塑物流行业格局"
    violations = lint(
        "## 一、总则\n\n数字化转型正在重塑物流行业格局。[m-h-1]",
        "本科",
        materials=[_material("m-h-1", excerpt=excerpt)],
    )
    assert "unmarked_derived_content" not in _rules(violations)


def test_照抄守卫_摘录过短无特征_不触发() -> None:
    violations = lint(
        "## 一、总则\n\n培养高素质人才。",
        "本科",
        materials=[_material("m-h-1", excerpt="人才")],
    )
    assert "unmarked_derived_content" not in _rules(violations)


def test_违规模型_默认严重级别为error() -> None:
    violation = Violation(rule="示例", message="示例消息")
    assert violation.severity == "error"
