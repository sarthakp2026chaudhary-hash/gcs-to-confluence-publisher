# extract_logs_table.py

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from google.cloud import bigquery

from config import LogsConfig

logger = logging.getLogger(__name__)

ERROR_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "permission_auth": (
        re.compile(r"permission denied|403|unauthorized|forbidden", re.IGNORECASE),
        re.compile(r"auth|credential|token", re.IGNORECASE),
    ),
    "quota_rate_limit": (
        re.compile(r"quota|rate limit|429|too many requests", re.IGNORECASE),
    ),
    "not_found": (
        re.compile(r"not found|404|no such file", re.IGNORECASE),
    ),
    "timeout": (
        re.compile(r"timeout|timed out|deadline exceeded", re.IGNORECASE),
    ),
    "oom_memory": (
        re.compile(r"out of memory|oom|memoryerror|killed process", re.IGNORECASE),
    ),
    "network_connection": (
        re.compile(r"connection reset|connection refused|broken pipe|dns", re.IGNORECASE),
    ),
    "other": (
        re.compile(r"error|exception|traceback", re.IGNORECASE),
    ),
}

KEYWORD_PATTERNS: dict[str, re.Pattern[str]] = {
    "retrying": re.compile(r"retrying", re.IGNORECASE),
    "killed": re.compile(r"killed", re.IGNORECASE),
    "sigterm": re.compile(r"sigterm", re.IGNORECASE),
    "broken_dag": re.compile(r"broken dag", re.IGNORECASE),
    "import_error": re.compile(r"importerror|import error", re.IGNORECASE),
}

STACKTRACE_PATTERN = re.compile(r"Traceback \(most recent cal")


@dataclass
class ParsedLogSignals:
    error_counts: dict[str, int]
    keyword_counts: dict[str, int]
    stacktrace_seen: int
    tail_summary: str


class ErrorLogTableExtractor:
    """Reads and parses error log records from BigQuery."""

    def __init__(self, logs_config: LogsConfig):
        self.config = logs_config
        if not logs_config.error_log_table:
            raise ValueError("JONATHON_ERROR_LOG_TABLE must be set in config.")
        self.client = bigquery.Client()
        self.table_name = logs_config.error_log_table

    @staticmethod
    def _format_bigquery_literal(value: Any) -> str:
        if value is None:
            return "NULL"
        if isinstance(value, pd.Timestamp):
            timestamp = value.to_pydatetime()
            if timestamp.tzinfo is None:
                return f"TIMESTAMP('{timestamp.isoformat(sep=' ')}')"
            return f"TIMESTAMP('{timestamp.isoformat()}')"
        if isinstance(value, str):
            return "'" + value.replace("'", "''") + "'"
        return str(value)

    def _build_in_clause(self, column_name: str, values: list[Any], chunk_size: int = 500) -> str | None:
        cleaned_values = [value for value in values if not pd.isna(value)]

        if not cleaned_values:
            return None

        unique_values = list(dict.fromkeys(cleaned_values))

        chunks: list[str] = []
        for idx in range(0, len(unique_values), chunk_size):
            chunk = unique_values[idx: idx + chunk_size]
            formatted = ", ".join(self._format_bigquery_literal(value) for value in chunk)
            chunks.append(f"{column_name} IN ({formatted})")

        if len(chunks) == 1:
            return chunks[0]
        return "(" + " OR ".join(chunks) + ")"

    def _load_log_rows(self, dag_run_df: pd.DataFrame) -> pd.DataFrame:
        required_cols = {"dag_id", "run_id", "logical_date"}
        if not required_cols.issubset(dag_run_df.columns):
            logger.warning("dag_run_df missing required columns for log extraction.")
            return pd.DataFrame()

        run_keys = dag_run_df[["dag_id", "run_id", "logical_date"]].dropna(subset=["dag_id", "run_id"]).drop_duplicates().copy()

        if run_keys.empty:
            return pd.DataFrame()

        dag_clause = self._build_in_clause("dag_id", run_keys["dag_id"].astype(str).tolist())
        run_clause = self._build_in_clause("run_id", run_keys["run_id"].astype(str).tolist())

        if not dag_clause or not run_clause:
            return pd.DataFrame()

        where_parts = [dag_clause, run_clause]

        logical_dates = pd.to_datetime(run_keys["logical_date"], utc=True, errors="coerce").dropna()
        if not logical_dates.empty:
            min_ts = logical_dates.min() - pd.Timedelta(days=2)
            max_ts = logical_dates.max() + pd.Timedelta(days=2)
            where_parts.append(
                f"log_ts BETWEEN {self._format_bigquery_literal(min_ts)} AND {self._format_bigquery_literal(max_ts)}"
            )

        sql = f"""
SELECT
    dag_id,
    run_id,
    task_id,
    try_number,
    log_ts,
    severity,
    source_loc,
    message,
    full_block
FROM `{self.table_name}`
WHERE {' AND '.join(where_parts)}
ORDER BY dag_id, run_id, log_ts
""".strip()

        try:
            logger.info("Running BigQuery query for error log table %s", self.table_name)
            frame = self.client.query(sql).to_dataframe()
        except Exception as exc:
            logger.warning("Error loading error log table %s: %s", self.table_name, exc)
            return pd.DataFrame()

        if frame.empty:
            return frame

        if "log_ts" in frame.columns:
            frame["log_ts"] = pd.to_datetime(frame["log_ts"], utc=True, errors="coerce")

        frame = frame.merge(run_keys, on=["dag_id", "run_id"], how="inner")

        return frame

    @staticmethod
    def _coalesce_log_text(row: pd.Series) -> str:
        full_block = row.get("full_block")
        if pd.notna(full_block) and str(full_block).strip():
            return str(full_block)
        message = row.get("message")
        if pd.notna(message):
            return str(message)
        return ""

    def parse_log_text(self, text: str, tail_lines: int = 20) -> ParsedLogSignals:
        error_counts = {key: 0 for key in ERROR_PATTERNS}
        keyword_counts = {key: 0 for key in KEYWORD_PATTERNS}

        for line in text.splitlines():
            for category, patterns in ERROR_PATTERNS.items():
                if any(pattern.search(line) for pattern in patterns):
                    error_counts[category] += 1
            for key, pattern in KEYWORD_PATTERNS.items():
                if pattern.search(line):
                    keyword_counts[key] += 1

        stacktrace_seen = int(bool(STACKTRACE_PATTERN.search(text)))
        tail_summary = "\n".join(text.splitlines()[-tail_lines:])

        return ParsedLogSignals(
            error_counts=error_counts,
            keyword_counts=keyword_counts,
            stacktrace_seen=stacktrace_seen,
            tail_summary=tail_summary,
        )

    def extract_for_runs(
        self,
        dag_run_df: pd.DataFrame,
        task_instance_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Aggregate error-log-derived signals at DAG-run level."""

        log_rows = self._load_log_rows(dag_run_df)

        if log_rows.empty:
            return pd.DataFrame()

        rows: list[dict[str, object]] = []

        for dag_run, group in log_rows.groupby(["dag_id", "run_id", "logical_date"], dropna=False):
            dag_id, run_id, logical_date = dag_run

            agg_error_counts = {f"error_cat_{k}": 0 for k in ERROR_PATTERNS}
            agg_keyword_counts = {f"kw_{k}": 0 for k in KEYWORD_PATTERNS}
            stacktrace_seen_total = 0
            parsed_logs = 0
            tails: list[str] = []

            for _, log_row in group.head(self.config.max_objects_per_dag_run).iterrows():
                text = self._coalesce_log_text(log_row)
                if not text.strip():
                    continue

                try:
                    signals = self.parse_log_text(text)
                except Exception as exc:
                    logger.warning("Log read/parse failure for %s/%s: %s", dag_id, run_id, exc)
                    continue

                parsed_logs += 1
                stacktrace_seen_total += signals.stacktrace_seen
                tails.append(signals.tail_summary)

                for k, v in signals.error_counts.items():
                    agg_error_counts[f"error_cat_{k}"] += v
                for k, v in signals.keyword_counts.items():
                    agg_keyword_counts[f"kw_{k}"] += v

            if parsed_logs == 0:
                continue

            row: dict[str, object] = {
                "dag_id": dag_id,
                "run_id": run_id,
                "logical_date": logical_date,
                "log_objects_parsed": parsed_logs,
                "stacktrace_seen_count": stacktrace_seen_total,
                "stacktrace_seen_rate": stacktrace_seen_total / max(parsed_logs, 1),
                "last_n_lines_summary": "\n\n---\n\n".join(tails[-3:]),
            }
            row.update(agg_error_counts)
            row.update(agg_keyword_counts)
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        result = pd.DataFrame(rows)
        if "logical_date" in result.columns:
            result["logical_date"] = pd.to_datetime(result["logical_date"], utc=True, errors="coerce")

        for col in result.columns:
            if col.startswith("error_cat_") or col.startswith("kw_"):
                result[f"{col}_rate"] = result[col] / result["log_objects_parsed"].replace({0: 1})

        return result
