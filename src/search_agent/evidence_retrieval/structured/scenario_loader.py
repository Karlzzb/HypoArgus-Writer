"""Load and validate the audited, read-only Doris scenario templates."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .scenario_registry import SCENARIO_REGISTRY


_SCENARIO_FILES = {
    "scenario_1": "query_coop_enterprise_qualification.yaml",
    "scenario_2": "query_enterprise_intern_student_count.yaml",
    "scenario_3": "query_hightech_vs_normal_salary_compare.yaml",
    "scenario_4": "query_major_internship_salary_stats.yaml",
    "scenario_5": "query_major_matched_position_trend.yaml",
    "scenario_6": "query_peer_school_major_gap.yaml",
    "scenario_7": "query_position_skill_tag_atlas.yaml",
    "scenario_8": "query_region_industry_job_demand.yaml",
    "scenario_9": "query_region_industry_salary_heatmap.yaml",
    "scenario_10": "query_student_attendance.yaml",
    "scenario_11": "query_student_job.yaml",
    "scenario_12": "query_student_origin_intern_city.yaml",
}
_PARAM = re.compile(r":([A-Za-z_][A-Za-z0-9_]*)")
_QUALIFIED_TABLE = re.compile(
    r"\b(?:FROM|JOIN)\s+`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*`?([A-Za-z_][A-Za-z0-9_]*)`?",
    re.IGNORECASE,
)
_MUTATING_SQL = re.compile(
    r"^\s*(?:INSERT|UPDATE|DELETE|REPLACE|UPSERT|DROP|ALTER|CREATE|TRUNCATE|LOAD|EXPORT|GRANT|REVOKE)\b",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class ScenarioTemplate:
    scenario_key: str
    name: str
    description: str
    sql: str
    params: dict[str, dict[str, Any]]
    return_columns: tuple[str, ...]
    limit: int

    def bind_params(self, supplied: dict[str, Any]) -> dict[str, Any]:
        unknown = set(supplied) - set(self.params)
        if unknown:
            raise ValueError(f"unknown parameters: {', '.join(sorted(unknown))}")
        bound: dict[str, Any] = {}
        for name, rule in self.params.items():
            if name in supplied:
                value = supplied[name]
            elif "default" in rule:
                value = rule["default"]
            elif rule.get("required"):
                raise ValueError(f"missing required parameter: {name}")
            else:
                value = None
            if rule.get("wildcard") == "like" and isinstance(value, str) and value:
                value = value if value.startswith("%") or value.endswith("%") else f"%{value}%"
            bound[name] = value
        return bound


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"invalid YAML mapping: {path.name}")
    return raw


def _table_registry(root: Path) -> dict[str, str]:
    raw = _read_yaml(root / "table_registry.yaml")
    tables = raw.get("tables")
    if not isinstance(tables, dict):
        raise ValueError("table_registry.yaml must define a tables mapping")
    return {str(table): str(database) for table, database in tables.items()}


def _validate_sql(sql: str, tables: dict[str, str], filename: str) -> None:
    clean = sql.strip()
    if not re.match(r"^(?:SELECT|WITH)\b", clean, re.IGNORECASE):
        raise ValueError(f"{filename}: only SELECT/WITH is allowed")
    without_tail = clean[:-1] if clean.endswith(";") else clean
    if ";" in without_tail or _MUTATING_SQL.search(without_tail):
        raise ValueError(f"{filename}: mutating or multi-statement SQL is forbidden")
    for database, table in _QUALIFIED_TABLE.findall(without_tail):
        expected = tables.get(table)
        if expected is None or expected != database:
            raise ValueError(f"{filename}: unregistered table reference {database}.{table}")


@lru_cache(maxsize=1)
def load_scenario_templates() -> dict[str, ScenarioTemplate]:
    root = Path(__file__).resolve().parent
    tables = _table_registry(root)
    templates: dict[str, ScenarioTemplate] = {}
    for scenario_key, filename in _SCENARIO_FILES.items():
        raw = _read_yaml(root / "scenarios" / filename)
        sql = str(raw.get("sql") or "")
        _validate_sql(sql, tables, filename)
        params = raw.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError(f"{filename}: params must be a mapping")
        normalized_params = {
            str(name): dict(rule) if isinstance(rule, dict) else {}
            for name, rule in params.items()
        }
        declared_params = set(SCENARIO_REGISTRY[scenario_key].get("params") or {})
        if set(normalized_params) != declared_params or set(_PARAM.findall(sql)) != declared_params:
            raise ValueError(f"{filename}: SQL/YAML/SearchAgent parameter contract mismatch")
        returned = tuple(
            str(column.get("name"))
            for column in (raw.get("return") or {}).get("columns", [])
            if isinstance(column, dict) and column.get("name")
        )
        expected_returns = tuple(SCENARIO_REGISTRY[scenario_key].get("return_columns") or ())
        if returned != expected_returns:
            raise ValueError(f"{filename}: return columns do not match SearchAgent contract")
        templates[scenario_key] = ScenarioTemplate(
            scenario_key=scenario_key,
            name=str(raw.get("name") or scenario_key),
            description=str(raw.get("description") or ""),
            sql=sql.strip().rstrip(";"),
            params=normalized_params,
            return_columns=returned,
            limit=max(1, int(raw.get("limit") or 20)),
        )
    return templates


__all__ = ["ScenarioTemplate", "load_scenario_templates"]
