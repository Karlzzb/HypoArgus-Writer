"""Read-only registry for the 12 audited structured business scenarios."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any


SCENARIO_REGISTRY: dict[str, dict[str, Any]] = {
    "scenario_1": {
        "scenario_name": "校企合作企业资质背景审查",
        "description": "按标签类型筛选合作企业，输出企业名称、统一社会信用代码、登记状态、标签清单、成立年限及经营异常红牌标记",
        "keywords": ["企业资质", "合作企业", "资质背景", "背景审查", "社会信用代码", "红牌", "经营异常"],
        "params": {
            "label_type": {"type": "string", "required": False, "default": "",
                           "description": "企业标签类型过滤，如'产教融合型企业'/'上市企业'/'高新技术企业'"}
        },
        "return_columns": ["ent_name", "credit_code", "ent_status", "ent_type_value", "ent_size", "province_name",
                           "city", "label_list", "es_date", "established_years", "red_flag"]
    },

    "scenario_2": {
        "scenario_name": "统计某企业接收本校实习生人数查询",
        "description": "统计指定企业按学年接收本校实习生的人数、累计人数及涉及专业分布",
        "keywords": ["接收实习生", "实习人数", "企业接收", "专业分布", "接收专业"],
        "params": {
            "enterprise_name": {"type": "string", "required": True, "wildcard": "like",
                                "description": "企业名称，支持模糊匹配"},
            "school_id": {"type": "string", "required": False, "default": "",
                          "description": "本校 school_id 过滤；留空表示全平台统计"}
        },
        "return_columns": ["enterprise_name", "academic_year", "intern_count", "major_count", "major_distribution"]
    },

    "scenario_3": {
        "scenario_name": "高新技术企业与其他企业实习薪资差异分析",
        "description": "对比命中指定企业标签（默认'高新技术企业'）与未命中企业发布的实习岗位薪资分布，输出双组薪资统计",
        "keywords": ["薪资差异", "高新技术企业", "薪资对比", "薪资分布", "中位数", "分位数"],
        "params": {
            "label_type": {"type": "string", "required": False, "default": "高新技术企业",
                           "description": "目标组企业标签类型，默认'高新技术企业'"}
        },
        "return_columns": ["group_label", "sample_count", "company_count", "avg_salary", "median_salary", "p25_salary",
                           "p75_salary", "max_salary", "min_salary"]
    },

    "scenario_4": {
        "scenario_name": "各专业毕业生实习薪资质量评估",
        "description": "按专业与学年统计学生实习岗位薪资的平均/最大/最小/中位数/P25/P75，剔除薪资为0或超出阈值的异常记录",
        "keywords": ["薪资质量", "专业薪资", "毕业生薪资", "质量评估", "薪资阈值"],
        "params": {
            "major_name": {"type": "string", "required": True, "wildcard": "like",
                           "description": "专业名称，支持模糊匹配"},
            "start_year": {"type": "string", "required": True, "description": "起始学年（年级），如 '2020级'"},
            "end_year": {"type": "string", "required": True, "description": "结束学年（年级），如 '2024级'"},
            "salary_upper_bound": {"type": "number", "required": False, "default": 100000,
                                   "description": "薪资上限阈值，用于剔除异常高薪记录"}
        },
        "return_columns": ["major_name", "academic_year", "sample_count", "avg_salary", "max_salary", "min_salary",
                           "median_salary", "p25_salary", "p75_salary"]
    },

    "scenario_5": {
        "scenario_name": "某专业对口岗位近N个月招聘趋势",
        "description": "根据专业名称定位推荐对口岗位，按月统计招聘发布数与平均薪资，输出月度趋势",
        "keywords": ["招聘趋势", "对口岗位", "趋势分析", "发布数", "月度趋势"],
        "params": {
            "major_name": {"type": "string", "required": True, "wildcard": "like",
                           "description": "专业名称，支持模糊匹配"},
            "start_date": {"type": "string", "required": True, "description": "趋势起始日期，格式 'YYYY-MM-DD'"},
            "end_date": {"type": "string", "required": True, "description": "趋势结束日期（左开区间）"}
        },
        "return_columns": ["publish_month", "job_name", "job_publish_count", "avg_salary"]
    },

    "scenario_6": {
        "scenario_name": "对标院校专业设置缺口分析",
        "description": "找出同类型院校已开设但本校尚未设置的专业，按开设院校数排序，输出缺口专业推荐列表",
        "keywords": ["专业缺口", "对标院校", "设置缺口", "尚未设置", "缺失专业"],
        "params": {
            "my_school_id": {"type": "string", "required": True, "description": "本校 school_id"},
            "peer_level": {"type": "string", "required": False, "default": "", "description": "对标院校层次过滤"},
            "peer_double": {"type": "string", "required": False, "default": "",
                            "description": "对标院校是否双高校（0/1）"}
        },
        "return_columns": ["gap_major_name", "peer_school_count", "peer_school_examples"]
    },

    "scenario_7": {
        "scenario_name": "特定岗位类型技能标签图谱分析",
        "description": "按岗位类型筛选岗位，提取关联技能标签并按频次排序，输出技能词云与共现分析底表",
        "keywords": ["技能标签", "技能图谱", "词云", "共现分析", "岗位技能", "特征字段"],
        "params": {
            "position_name": {"type": "string", "required": True, "wildcard": "like",
                              "description": "岗位类型关键词，支持模糊匹配"},
            "start_date": {"type": "string", "required": True, "description": "岗位发布起始日期，格式 'YYYY-MM-DD'"},
            "end_date": {"type": "string", "required": True, "description": "岗位发布结束日期（左开区间）"}
        },
        "return_columns": ["feature_name", "feature_category", "job_count"]
    },

    "scenario_8": {
        "scenario_name": "区域产业人才需求热力图",
        "description": "按城市×行业大类统计岗位发布数量，支持按省/市和发布日期范围过滤，输出热力矩阵数据",
        "keywords": ["人才需求", "需求热力", "热力图", "行业大类", "发布数量", "区域产业"],
        "params": {
            "start_date": {"type": "string", "required": True, "description": "岗位发布起始日期 'YYYY-MM-DD'"},
            "end_date": {"type": "string", "required": True, "description": "岗位发布结束日期（左开区间）"},
            "province": {"type": "string", "required": False, "default": "", "description": "省份过滤"},
            "city": {"type": "string", "required": False, "default": "", "description": "城市过滤"}
        },
        "return_columns": ["province", "city", "industry_level1", "job_publish_count", "company_count"]
    },

    "scenario_9": {
        "scenario_name": "区域行业薪资热力指数",
        "description": "构建城市×行业二维薪资指数，输出各网格薪资中位数及相对基准的指数，支持省/市和日期过滤",
        "keywords": ["薪资热力", "薪资指数", "相对指数", "中位数", "基准"],
        "params": {
            "start_date": {"type": "string", "required": True, "description": "岗位发布起始日期 'YYYY-MM-DD'"},
            "end_date": {"type": "string", "required": True, "description": "岗位发布结束日期（左开区间）"},
            "province": {"type": "string", "required": False, "default": "", "description": "省份过滤"},
            "city": {"type": "string", "required": False, "default": "", "description": "城市过滤"},
            "min_sample": {"type": "number", "required": False, "default": 5, "description": "网格最小样本数阈值"}
        },
        "return_columns": ["city", "industry_level1", "sample_count", "median_salary", "salary_index"]
    },

    "scenario_10": {
        "scenario_name": "查询某专业学生实习岗位分布",
        "description": "按专业名称统计学生在岗实习的岗位分布（岗位名称、企业、人数）",
        "keywords": ["实习岗位", "岗位分布", "在岗实习", "学生人数"],
        "params": {
            "major_name": {"type": "string", "required": True, "wildcard": "like",
                           "description": "专业名称，支持模糊匹配"}
        },
        "return_columns": ["job_name", "company_name", "student_count"]
    },

    "scenario_11": {
        "scenario_name": "按学年统计某专业下的学生人数，并计算同比增长率",
        "description": "按学年统计某专业下的学生人数，并计算同比增长率（通过专业名称过滤）",
        "keywords": ["学生人数", "同比增长", "增长率", "学年人数"],
        "params": {
            "major_name": {"type": "string", "required": True, "wildcard": "like",
                           "description": "专业名称，支持模糊匹配"},
            "start_year": {"type": "string", "required": True, "description": "起始学年，如 '2020'"},
            "end_year": {"type": "string", "required": True, "description": "结束学年，如 '2024'"}
        },
        "return_columns": ["academic_year", "student_count", "prev_year_count", "growth_rate_percent"]
    },

    "scenario_12": {
        "scenario_name": "学生生源地与实习城市分布及就业基地指导",
        "description": "统计学生生源地 TOP10、实习城市 TOP10 及生源地回流率，辅助就业基地布局决策",
        "keywords": ["生源地", "实习城市", "回流率", "TOP10", "就业基地"],
        "params": {
            "school_id": {"type": "string", "required": False, "default": "", "description": "学校 school_id 过滤"},
            "grade": {"type": "string", "required": False, "default": "", "description": "年级过滤"},
            "major_name": {"type": "string", "required": False, "default": "", "wildcard": "like",
                           "description": "专业名称过滤，支持模糊匹配"}
        },
        "return_columns": ["metric_type", "dim_value", "cnt"]
    }
}

STRUCTURED_SCENARIOS = MappingProxyType(SCENARIO_REGISTRY)
