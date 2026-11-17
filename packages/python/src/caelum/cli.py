"""Command line: ``caelum cache stats`` (also ``python -m caelum cache stats``).

Reads the SQLite cache file and prints what it holds: entry counts (exact and
semantic), expired entries, size on disk, and the estimated cost of the cached answers,
i.e. what one more reuse of every entry would save. It never modifies the file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from caelum.cache.stores import default_cache_path
from caelum.pricing import estimate_cost
from caelum.utils import extract_usage, format_usd, served_model

__all__ = ["CacheStats", "ModelStats", "cache_stats", "main"]


@dataclass
class ModelStats:
    """Live entries answered by one model."""

    entries: int = 0
    value_usd: float = 0.0
    unpriced: int = 0


@dataclass
class CacheStats:
    """What a CAELUM cache file holds."""

    path: str
    size_bytes: int
    exact_entries: int = 0
    exact_expired: int = 0
    semantic_entries: int = 0
    semantic_expired: int = 0
    oldest: float | None = None
    newest: float | None = None
    value_usd: float = 0.0
    unpriced_entries: int = 0
    models: dict[str, ModelStats] = field(default_factory=lambda: {})

    @property
    def live_entries(self) -> int:
        """Entries that can still be served."""
        return (
            self.exact_entries - self.exact_expired + self.semantic_entries - self.semantic_expired
        )


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        items = cast("dict[str, Any]", value).items()
        return SimpleNamespace(**{str(k): _namespace(v) for k, v in items})
    return value


def _answer_cost(raw: str) -> tuple[str, float | None]:
    """(model, estimated cost) of one cached answer."""
    try:
        payload = json.loads(raw)
    except ValueError:
        return "unknown", None
    if not isinstance(payload, dict):
        return "unknown", None
    response = _namespace(payload)
    model = served_model(response) or "unknown"
    usage = extract_usage(response)
    if usage is None:
        return model, None
    return model, estimate_cost(model, usage[0], usage[1])


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def _file_size(path: Path) -> int:
    # WHY: in WAL mode recent writes live in the -wal file until a checkpoint.
    total = 0
    for suffix in ("", "-wal", "-shm"):
        part = Path(f"{path}{suffix}")
        if part.exists():
            total += part.stat().st_size
    return total


def cache_stats(
    path: str | os.PathLike[str] | None = None, *, now: float | None = None
) -> CacheStats:
    """Read statistics from a CAELUM cache file (read-only).

    Args:
        path: Cache file (default ``~/.caelum/cache.sqlite`` or ``$CAELUM_HOME``).
        now: Current time (for tests).

    Raises:
        FileNotFoundError: The file does not exist.
        ValueError: The file is not a CAELUM cache.
    """
    target = Path(path) if path is not None else default_cache_path()
    if not target.is_file():
        raise FileNotFoundError(f"no cache file at {target}")
    current = time.time() if now is None else now
    stats = CacheStats(path=str(target), size_bytes=_file_size(target))
    try:
        conn = sqlite3.connect(f"{target.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as exc:  # pragma: no cover - permissions, locked media...
        raise ValueError(f"cannot open {target}: {exc}") from exc
    try:
        tables = [t for t in ("caelum_cache", "caelum_semantic") if _has_table(conn, t)]
        if not tables:
            raise ValueError(f"{target} is not a CAELUM cache file")
        for table in tables:
            rows = conn.execute(f"SELECT value, created_at, expires_at FROM {table}")  # noqa: S608
            for value, created_at, expires_at in rows:
                # WHY: a hand-edited or foreign row must not crash the report.
                expired = isinstance(expires_at, (int, float)) and expires_at <= current
                if table == "caelum_cache":
                    stats.exact_entries += 1
                    stats.exact_expired += expired
                else:
                    stats.semantic_entries += 1
                    stats.semantic_expired += expired
                if expired:
                    continue
                stats.oldest = created_at if stats.oldest is None else min(stats.oldest, created_at)
                stats.newest = created_at if stats.newest is None else max(stats.newest, created_at)
                model, cost = _answer_cost(value)
                per_model = stats.models.setdefault(model, ModelStats())
                per_model.entries += 1
                if cost is None:
                    per_model.unpriced += 1
                    stats.unpriced_entries += 1
                else:
                    per_model.value_usd += cost
                    stats.value_usd += cost
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"{target} is not a readable CAELUM cache file: {exc}") from exc
    finally:
        conn.close()
    return stats


def _size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"  # pragma: no cover


def _date(ts: float | None) -> str:
    if ts is None:
        return "-"
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def format_stats(stats: CacheStats) -> str:
    """Human-readable report."""
    lines = [
        "CAELUM cache",
        f"  File:              {stats.path}",
        f"  Size on disk:      {_size(stats.size_bytes)}",
        f"  Exact entries:     {stats.exact_entries} ({stats.exact_expired} expired)",
        f"  Semantic entries:  {stats.semantic_entries} ({stats.semantic_expired} expired)",
        f"  Live entries:      {stats.live_entries}",
        f"  Oldest / newest:   {_date(stats.oldest)} / {_date(stats.newest)}",
        f"  Value per reuse:   {format_usd(stats.value_usd)}"
        "  (estimated cost of one more call for every live entry)",
    ]
    if stats.unpriced_entries:
        lines.append(f"  Unpriced entries:  {stats.unpriced_entries} (model price unknown)")
    if stats.models:
        lines.append("")
        lines.append("  By model:")
        width = max(len(name) for name in stats.models)
        ranked = sorted(stats.models.items(), key=lambda item: (-item[1].entries, item[0]))
        for name, model in ranked:
            priced = model.unpriced < model.entries
            value = format_usd(model.value_usd) if priced else "price unknown"
            noun = "entry" if model.entries == 1 else "entries"
            lines.append(f"    {name.ljust(width)}  {model.entries:>6} {noun:<7}  {value}")
    lines.append("")
    lines.append("Savings over time are counted per client: see client.metrics.print_summary().")
    return "\n".join(lines)


def _stats_command(args: argparse.Namespace) -> int:
    try:
        stats = cache_stats(args.path)
    except FileNotFoundError as exc:
        print(f"caelum: {exc} (nothing cached yet, or pass --path)", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"caelum: {exc}", file=sys.stderr)
        return 1
    if args.json:
        data = asdict(stats)
        data["live_entries"] = stats.live_entries
        print(json.dumps(data, indent=2))
    else:
        print(format_stats(stats))
    return 0


def _cache_clear(args: argparse.Namespace) -> int:
    """Delete one end customer's cached answers (local file, and Cloud with a key)."""
    from caelum.cache.stores import SQLiteStore, default_cache_path
    from caelum.cache.vectors import SQLiteVectorStore
    from caelum.cloud import CloudConnection, customer_tag, resolve_cloud_key
    from caelum.journal import Journal, JournalEntry

    older_than = args.older_than
    if older_than is not None and older_than <= 0:
        print("caelum: --older-than doit être un nombre de secondes positif", file=sys.stderr)
        return 1
    path = Path(args.path) if args.path else default_cache_path()
    local = 0
    if path.exists():
        store = SQLiteStore(path)
        vectors = SQLiteVectorStore(path)
        try:
            local = store.clear_customer(args.customer, older_than)
            local += vectors.delete_customer(args.customer, older_than)
        finally:
            store.close()
            vectors.close()
    cloud: int | None = None
    key = resolve_cloud_key(None)
    if key is not None:
        connection = CloudConnection(key)
        try:
            cloud = connection.cache_clear(customer_tag(args.customer), older_than)
        finally:
            connection.close()
    journal = Journal()
    try:
        journal.record(
            JournalEntry(
                outcome="purge",
                customer=args.customer,
                reason="clear" if older_than is None else f"older_than:{int(older_than)}s",
                entry_id=f"local:{local} cloud:{'-' if cloud is None else cloud}",
            )
        )
    finally:
        journal.close()
    who = f"customer {args.customer!r}" if args.customer else "requests without a customer"
    where = "" if cloud is None else f", {cloud} in CAELUM Cloud"
    print(f"caelum: {local} cached answer(s) of {who} deleted locally{where}")
    return 0


def _journal_clear(args: argparse.Namespace) -> int:
    from caelum.journal import clear_journal_file

    erased = clear_journal_file(args.path, args.customer)
    who = f" of customer {args.customer!r}" if args.customer else ""
    print(f"caelum: {erased} journal line(s){who} erased")
    return 0


def _journal_stats(args: argparse.Namespace) -> int:
    from caelum.journal import Journal, default_journal_path

    path = Path(args.path) if args.path else default_journal_path()
    if not path.exists():
        print(f"caelum: no journal at {path}", file=sys.stderr)
        return 1
    journal = Journal(path)
    try:
        since = time.time() - args.days * 86_400 if args.days else None
        print(json.dumps(journal.stats(args.customer, since=since), indent=2, ensure_ascii=False))
    finally:
        journal.close()
    return 0


def _parser() -> argparse.ArgumentParser:
    from caelum import __version__

    parser = argparse.ArgumentParser(prog="caelum", description="CAELUM command line tools.")
    parser.add_argument("--version", action="version", version=f"caelum {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("cache", help="inspect the local cache")
    cache_commands = cache.add_subparsers(dest="cache_command", required=True)
    stats = cache_commands.add_parser("stats", help="what the cache file holds")
    stats.add_argument("--path", help="cache file (default: ~/.caelum/cache.sqlite)")
    stats.add_argument("--json", action="store_true", help="machine-readable output")
    stats.set_defaults(handler=_stats_command)
    cclear = cache_commands.add_parser(
        "clear", help="delete one end customer's cached answers (local, and Cloud with a key)"
    )
    cclear.add_argument("--customer", default="", help="end customer (default: none)")
    cclear.add_argument(
        "--older-than", type=float, help="only answers cached more than N seconds ago"
    )
    cclear.add_argument("--path", help="cache file (default: ~/.caelum/cache.sqlite)")
    cclear.set_defaults(handler=_cache_clear)
    journal = commands.add_parser("journal", help="the local per-request journal")
    journal_commands = journal.add_subparsers(dest="journal_command", required=True)
    clear = journal_commands.add_parser("clear", help="erase the journal")
    clear.add_argument("--path", help="journal file (default: ~/.caelum/journal.sqlite)")
    clear.add_argument("--customer", help="erase only this end customer's lines")
    clear.set_defaults(handler=_journal_clear)
    jstats = journal_commands.add_parser("stats", help="statistics of one end customer")
    jstats.add_argument("--path", help="journal file (default: ~/.caelum/journal.sqlite)")
    jstats.add_argument("--customer", help="end customer (default: requests without one)")
    jstats.add_argument("--days", type=float, help="only the last N days")
    jstats.set_defaults(handler=_journal_stats)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``caelum`` command."""
    args = _parser().parse_args(argv)
    return int(args.handler(args))
