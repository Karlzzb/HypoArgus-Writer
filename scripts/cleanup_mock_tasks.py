"""生产清理演练脚本：批量删除 mock- 前缀线程的检查点数据。

mock 栈线程可在线上任意触发（创建任务时 ``mock=true``），故需定期清理，
避免检查点表无限累积。本脚本直接对 Postgres 的三张检查点表跑批量
``DELETE``，按 ``thread_id LIKE 'mock-%'`` 过滤，并可按 ``metadata`` 中的
``created_at`` 早于 cutoff 进一步收窄（仅 ``checkpoints`` 表有 metadata 列）。

用法::

    uv run python scripts/cleanup_mock_tasks.py

环境变量：
- ``HYPOARGUS_PG_DSN``：必填，Postgres 连接串。
- ``MOCK_CLEANUP_DAYS``：选填，正整数，缺省 7；只删 metadata.created_at
  早于 ``now - N 天`` 的 mock- 线程。

已知风险：
- ``checkpoint_blobs`` 与 ``checkpoint_writes`` 无 metadata 列，本脚本只按
  ``thread_id LIKE 'mock-%'`` 删除这两表，不应用 cutoff——即所有 mock-
  前缀行的 blobs / writes 都会被清掉。这与 LangGraph 的线程删除语义一致
  （删线程即删其全部 blob 与 write）。
- 脚本不在事务里跨表串行化：每表独立 ``DELETE``，中途失败需手动重跑；
  PostgresSaver 已 autocommit，每条 DELETE 即时落盘。
- 不删内存登记：生产环境与该脚本进程不共享 TaskManager 实例，内存登记
  由各自进程的 TTL/关停自然回收。
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

from domain.env_config import read_positive_int
from graph import postgres_checkpointer


def _format_cutoff(days: int) -> datetime:
    """计算 cutoff 时刻（UTC now - days）。"""
    return datetime.now(timezone.utc) - timedelta(days=days)


def main() -> int:
    dsn = os.environ.get("HYPOARGUS_PG_DSN")
    if not dsn:
        print("错误：环境变量 HYPOARGUS_PG_DSN 未设置", file=sys.stderr)
        return 2

    days = read_positive_int(os.environ, "MOCK_CLEANUP_DAYS", 7)
    cutoff = _format_cutoff(days)
    cutoff_iso = cutoff.isoformat()

    print(f"mock 线程清理：cutoff = {cutoff_iso}（MOCK_CLEANUP_DAYS={days}）")

    with postgres_checkpointer(dsn=dsn) as saver:
        conn = saver.conn
        total = 0

        # checkpoints 表有 metadata 列，按 created_at 收窄。
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM checkpoints "
                "WHERE thread_id LIKE 'mock-%' "
                "AND (metadata->>'created_at')::timestamptz < %s",
                (cutoff_iso,),
            )
            deleted = cur.rowcount or 0
            print(f"checkpoints  删除 {deleted} 行")
            total += deleted

        # blobs / writes 无 metadata 列，只按前缀删（与删线程语义一致）。
        for table in ("checkpoint_blobs", "checkpoint_writes"):
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {table} WHERE thread_id LIKE 'mock-%'"  # noqa: S608 — 表名是字面量，非用户输入
                )
                deleted = cur.rowcount or 0
                print(f"{table}  删除 {deleted} 行")
                total += deleted

    print(f"总计删除 {total} 行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
