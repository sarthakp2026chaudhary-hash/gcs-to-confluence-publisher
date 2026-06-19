# rca.py
#
# Root Cause Analysis (RCA) module for DAG failure risk.
#
# Queries JONATHON_DAG_ERROR_LOG_RECORDS by dag_id (across past runs) and
# produces structured RCA summaries used by both predict.py (CSV enrichment)
# and report.py (human-readable sections).
#
# RCA signals extracted per DAG:
#   rca_error_category   : classified error type (ImportError, Timeout, Permission, etc.)
#   rca_top_task         : task_id that fails most often
#   rca_top_error        : most recent raw error message (short)
#   rca_stack_hint       : last meaningful line from full_block stack trace
#   rca_severity         : worst severity seen (CRITICAL > ERROR > WARNING)
#   rca_recurrence       : number of distinct run_ids with errors
#   rca_error_count      : total error records for the DAG
#   rca_try_number_max   : highest try_number seen (indicates persistent failure)
#   rca_first_seen       : earliest log_ts
#   rca_last_seen        : latest log_ts

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from google.cloud import bigquery

from config import LogsConfig

logger = logging.getLogger(__name__)

# --- Error classification ---

_SEVERITY_RANK = {"CRITICAL": 4, "ERROR": 3, "WARNING": 2, "INFO": 1}

_ERROR_CATEGORIES: list[tuple[str, re.Pattern[str]]] = [
    ("ImportError",       re.compile(r"importerror|modulenot|no module named|broken dag", re.I)),
    ("PermissionDenied",  re.compile(r"permission denied|403|401|unauthorized|forbidden|iam", re.I)),
    ("Timeout",           re.compile(r"timeout|timed out|deadline exceeded|operator timeout", re.I)),
    ("OOMMemory",         re.compile(r"out of memory|oom|memoryerror|killed|sigkill", re.I)),
    ("QuotaRateLimit",    re.compile(r"quota|rate.?limit|429|too many requests", re.I)),
    ("NetworkConnection", re.compile(r"connection reset|connection refused|broken pipe|dns|socket", re.I)),
    ("NotFound",          re.compile(r"not found|404|no such file|table.*not exist", re.I)),
    ("BigQueryError",     re.compile(r"bigquery|bq\s|job.*failed|query.*error", re.I)),
    ("CodeError",         re.compile(r"keyerror|attributeerror|typeerror|valueerror|assertionerror", re.I)),
    ("TaskFailed",        re.compile(r"task.*failed|upstream.*failed|marked.*failed", re.I)),
]

_STACKTRACE_NOISE = re.compile(
    r"^(\s*File |\s*Traceback|\s*at |\s*\.\.\.|\s*$)", re.M
)


def _classify_error(text: str) -> str:
    """Return first matching error category or 'Other'."""
    for category, pattern in _ERROR_CATEGORIES:
        if pattern.search(text):
            return category
    return "Other"


def _extract_stack_hint(full_block: str | None, max_len: int = 200) -> str:
    """Extract the most meaningful line from a stack trace.

    Prefers the last non-boilerplate line before 'Error:' or last non-empty line.
    """
    if not full_block or not str(full_block).strip():
        return ""
    lines = [ln.strip() for ln in str(full_block).splitlines() if ln.strip()]

    # Find last line starting with a known exception class
    for line in reversed(lines):
        if re.match(r"[A-Za-z][\w.]*Error|[A-Za-z][\w.]*Exception|[A-Za-z][\w.]*Warning", line):
            return line[:max_len]

    # Fall back to last non-empty line
    return lines[-1][:max_len] if lines else ""


def _worst_severity(severities: pd.Series) -> str:
    """Return highest severity from a series of severity strings."""
    best = "INFO"
    best_rank = 0
    for s in severities.dropna():
        rank = _SEVERITY_RANK.get(str(s).upper(), 0)
        if rank > best_rank:
            best_rank = rank
            best = str(s).upper()
    return best


# --- RCA data classes ---

@dataclass
class RCASummary:
    """Flat summary per dag_id — maps directly to CSV columns."""
    dag_id: str
    rca_error_category: str = "Unknown"
    rca_top_task: str = ""
    rca_top_error: str = ""
    rca_stack_hint: str = ""
    rca_severity: str = ""
    rca_recurrence: int = 0       # distinct run_ids with errors
    rca_error_count: int = 0      # total error records
    rca_try_number_max: int = 0   # highest try_number seen
    rca_first_seen: str = ""
    rca_last_seen: str = ""


@dataclass
class RCADetail:
    """Rich per-dag_id detail for report rendering (includes top-N error entries)."""
    summary: RCASummary
    top_entries: list[dict[str, Any]] = field(default_factory=list)  # top 5 most recent


# --- Main extractor ---

class RCAExtractor:
    """Queries JONATHON_DAG_ERROR_LOG_RECORDS and builds RCA summaries
    for a list of dag_ids.

    Designed to be used by both predict.py and report.py.
    """

    def __init__(self, logs_config: LogsConfig):
        if not logs_config.error_log_table:
            raise ValueError("JONATHON_ERROR_LOG_TABLE must be set for RCA extraction.")
        self.config = logs_config
        self.table_name = logs_config.error_log_table
        self.client = bigquery.Client()

    def _fetch_raw(self, dag_ids: list[str]) -> pd.DataFrame:
        """Fetch raw error log rows for given dag_ids."""
        if not dag_ids:
            return pd.DataFrame()

        escaped = ", ".join([f"'{d}'" for d in dag_ids if d and str(d) != "nan"])
        if not escaped:
            return pd.DataFrame()

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
WHERE dag_id IN ({escaped})
ORDER BY dag_id, log_ts DESC
LIMIT 50000
""".strip()

        try:
            logger.info("RCA: querying %s for %d dag_ids", self.table_name, len(dag_ids))
            df = self.client.query(sql).to_dataframe()
            if "log_ts" in df.columns:
                df["log_ts"] = pd.to_datetime(df["log_ts"], utc=True, errors="coerce")
            logger.info("RCA: fetched %d error log rows", len(df))
            return df
        except Exception as exc:
            logger.warning("RCA: BigQuery fetch failed: %s", exc)
            return pd.DataFrame()

    def _fetch_raw_for_failed_runs(self, dag_run_df: pd.DataFrame) -> pd.DataFrame:
        """Optimized fetch: query error logs only for DAGs that have failed runs.

        Filters JONATHON_DAG_ERROR_LOG_RECORDS by:
        - dag_ids that have state='failed' in the dag_run_df

        Returns ALL historical logs for those failed DAGs (not time-limited).
        This is more efficient than querying all logs for every dag_id.
        """
        if dag_run_df.empty:
            return pd.DataFrame()

        # Extract dag_ids that have failed/upstream_failed/timeout states
        failed_states = ["failed", "upstream_failed", "timeout"]
        failed_dag_ids = (
            dag_run_df[dag_run_df["state"].isin(failed_states)]["dag_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if not failed_dag_ids:
            logger.info("RCA: no failed DAGs found in run batch; skipping log fetch")
            return pd.DataFrame()

        # Build SQL IN clause for failed dag_ids
        dag_id_list = ", ".join([f"'{did}'" for did in failed_dag_ids])

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
WHERE dag_id IN ({dag_id_list})
ORDER BY dag_id, log_ts DESC
LIMIT 50000
""".strip()

        try:
            logger.info("RCA: querying %s for %d failed dag_ids (optimized fetch)", self.table_name, len(failed_dag_ids))
            df = self.client.query(sql).to_dataframe()
            if "log_ts" in df.columns:
                df["log_ts"] = pd.to_datetime(df["log_ts"], utc=True, errors="coerce")
            logger.info("RCA: fetched %d error log rows for failed DAGs", len(df))
            return df
        except Exception as exc:
            logger.warning("RCA: BigQuery optimized fetch failed: %s", exc)
            return pd.DataFrame()

    def _build_summary(self, dag_id: str, group: pd.DataFrame) -> RCASummary:
        """Build an RCASummary from a per-dag_id group of error log rows."""

        # Most recent row first (already sorted DESC from query)
        top_row = group.iloc[0]

        # Error category — try message first, then full_block
        combined_text = " ".join([
            str(top_row.get("message", "")),
            str(top_row.get("full_block", "") or ""),
        ])
        error_category = _classify_error(combined_text)

        # Top failing task (most frequent task_id in group)
        top_task = ""
        if "task_id" in group.columns:
            task_counts = group["task_id"].dropna().value_counts()
            if not task_counts.empty:
                top_task = str(task_counts.index[0])

        # Short error message
        top_error = str(top_row.get("message", ""))[:200]

        # Stack hint
        stack_hint = _extract_stack_hint(top_row.get("full_block"))

        # Worst severity across all rows
        severity = _worst_severity(group["severity"]) if "severity" in group.columns else "OK"

        # Recurrence: distinct run_ids
        recurrence = int(group["run_id"].nunique()) if "run_id" in group.columns else 0

        # Try number max
        try_max = int(group["try_number"].max()) if "try_number" in group.columns and not group["try_number"].isna().all() else 0

        # Time range
        first_seen = ""
        last_seen = ""
        if "log_ts" in group.columns:
            ts_clean = group["log_ts"].dropna()
            if not ts_clean.empty:
                first_seen = str(ts_clean.min())[:19]
                last_seen = str(ts_clean.max())[:19]

        return RCASummary(
            dag_id=dag_id,
            rca_error_category=error_category,
            rca_top_task=top_task,
            rca_top_error=top_error,
            rca_stack_hint=stack_hint,
            rca_severity=severity,
            rca_recurrence=recurrence,
            rca_error_count=len(group),
            rca_try_number_max=try_max,
            rca_first_seen=first_seen,
            rca_last_seen=last_seen,
        )

    def _build_detail(self, dag_id: str, group: pd.DataFrame) -> RCADetail:
        """Build full RCADetail (summary + top-5 entries) for report rendering."""
        summary = self._build_summary(dag_id, group)

        top_entries: list[dict[str, Any]] = []
        for _, row in group.head(5).iterrows():
            top_entries.append({
                "task_id": str(row.get("task_id", "")),
                "try_number": int(row.get("try_number", 0)) if pd.notna(row.get("try_number")) else 0,
                "log_ts": str(row.get("log_ts", ""))[:19],
                "severity": str(row.get("severity", "UNKNOWN")),
                "source_loc": str(row.get("source_loc", ""))[:100],
                "message": str(row.get("message", ""))[:200],
                "stack_hint": _extract_stack_hint(row.get("full_block")),
            })

        return RCADetail(summary=summary, top_entries=top_entries)

    def extract_summaries(self, dag_ids: list[str]) -> dict[str, RCASummary]:
        """Return flat RCASummary per dag_id.

        Used by predict.py to add RCA columns to predictions CSV.
        """
        raw = self._fetch_raw(dag_ids)
        if raw.empty:
            return {}

        result: dict[str, RCASummary] = {}
        for dag_id, group in raw.groupby("dag_id"):
            result[str(dag_id)] = self._build_summary(str(dag_id), group)

        logger.info("RCA summaries built for %d DAGs", len(result))
        return result

    def extract_details(self, dag_ids: list[str]) -> dict[str, RCADetail]:
        """Return RCADetail (summary + top entries) per dag_id.

        Used by report.py for human-readable RCA sections.
        """
        raw = self._fetch_raw(dag_ids)
        if raw.empty:
            return {}

        result: dict[str, RCADetail] = {}
        for dag_id, group in raw.groupby("dag_id"):
            result[str(dag_id)] = self._build_detail(str(dag_id), group)

        logger.info("RCA details built for %d DAGs", len(result))
        return result

    def extract_summaries_for_failed_runs(
        self,
        dag_run_df: pd.DataFrame,
        high_risk_dag_ids: list[str],
    ) -> dict[str, RCASummary]:
        """Optimized summaries: fetch logs only for DAGs that have failed states,
        then return summaries only for high-risk dag_ids.

        Args:
            dag_run_df:         DataFrame of DAG runs (contains state information)
            high_risk_dag_ids:  List of dag_ids marked as High-risk in predictions

        Returns:
            dict of {dag_id: RCASummary} for High-risk DAGs with available error logs
        """
        # Fetch logs only for DAGs with failed states (across all their history)
        raw = self._fetch_raw_for_failed_runs(dag_run_df)
        if raw.empty:
            return {}

        # Build summaries, but only return those requested
        result: dict[str, RCASummary] = {}
        for dag_id, group in raw.groupby("dag_id"):
            dag_id_str = str(dag_id)
            if dag_id_str in high_risk_dag_ids:
                result[dag_id_str] = self._build_summary(dag_id_str, group)

        logger.info(
            "RCA summaries built for %d of %d High-risk DAGs (logs only for failed DAGs)",
            len(result),
            len(set(high_risk_dag_ids)),
        )
        return result


# --- Enrichment helper for predict.py ---

# All columns added to predictions CSV
RCA_COLUMNS = [
    "rca_error_category",
    "rca_top_task",
    "rca_top_error",
    "rca_stack_hint",
    "rca_severity",
    "rca_recurrence",
    "rca_error_count",
    "rca_try_number_max",
    "rca_first_seen",
    "rca_last_seen",
]

_RCA_DEFAULTS: dict[str, Any] = {
    "rca_error_category": "",
    "rca_top_task": "",
    "rca_top_error": "",
    "rca_stack_hint": "",
    "rca_severity": "",
    "rca_recurrence": 0,
    "rca_error_count": 0,
    "rca_try_number_max": 0,
    "rca_first_seen": "",
    "rca_last_seen": "",
}


def enrich_with_rca(
    pred_df: pd.DataFrame,
    logs_config: LogsConfig,
    dag_run_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Enrich a predictions DataFrame with RCA columns for all High-risk DAGs.

    Non-high-risk rows get empty/zero defaults.

    Args:
        pred_df:      DataFrame with at minimum dag_id and risk_bucket columns.
        logs_config:  LogsConfig with error_log_table set.
        dag_run_df:   Optional DataFrame of DAG runs for optimized log fetching.
                      If provided, filters error logs to DAGs with failed states only.
                      If None, queries all historical logs per dag_id (slower).

    Returns:
        pred_df with RCA_COLUMNS appended.
    """
    # Set defaults for all rows
    for col, default in _RCA_DEFAULTS.items():
        pred_df[col] = default

    if not logs_config.enabled or not logs_config.error_log_table:
        logger.info("RCA enrichment skipped: log parsing disabled or table not configured.")
        return pred_df

    high_risk_dag_ids = (
        pred_df[pred_df["risk_bucket"] == "High"]["dag_id"]
        .dropna()
        .unique()
        .tolist()
    )

    if not high_risk_dag_ids:
        logger.info("RCA enrichment skipped: no High-risk DAGs in predictions.")
        return pred_df

    try:
        extractor = RCAExtractor(logs_config)

        # Use optimized fetch if dag_run_df is available
        if dag_run_df is not None and not dag_run_df.empty:
            summaries = extractor.extract_summaries_for_failed_runs(dag_run_df, high_risk_dag_ids)
        else:
            summaries = extractor.extract_summaries(high_risk_dag_ids)
    except Exception as exc:
        logger.warning("RCA enrichment failed, predictions will have empty RCA columns: %s", exc)
        return pred_df

    # Map RCA fields back to predictions by dag_id
    for col in RCA_COLUMNS:
        pred_df[col] = pred_df["dag_id"].apply(
            lambda d, c=col: getattr(summaries.get(d), c, _RCA_DEFAULTS[c])
            if d in summaries else _RCA_DEFAULTS[c]
        )

    logger.info("RCA enrichment complete for %d High-risk DAG(s)", len(summaries))
    return pred_df


# --- Report rendering helper for report.py ---

def build_rca_details_for_report(
    pred_df: pd.DataFrame,
    logs_config: LogsConfig,
) -> dict[str, RCADetail]:
    """Build RCADetail objects for High-risk DAGs to be rendered in the report.

    Args:
        pred_df:      Predictions DataFrame with dag_id and risk_bucket.
        logs_config:  LogsConfig with error_log_table set.

    Returns:
        dict of {dag_id: RCADetail}, empty dict if unavailable.
    """
    if not logs_config.enabled or not logs_config.error_log_table:
        return {}

    high_risk_dag_ids = (
        pred_df[pred_df["risk_bucket"] == "High"]["dag_id"]
        .dropna()
        .unique()
        .tolist()
    )

    if not high_risk_dag_ids:
        return {}

    try:
        extractor = RCAExtractor(logs_config)
        return extractor.extract_details(high_risk_dag_ids)
    except Exception as exc:
        logger.warning("RCA detail build failed for report: %s", exc)
        return {}


def render_rca_markdown(rca_details: dict[str, RCADetail]) -> str:
    """Render the RCA section as Markdown text for inclusion in the report.

    Args:
        rca_details: dict of {dag_id: RCADetail} from build_rca_details_for_report().

    Returns:
        Markdown string (empty string if no RCA data).
    """
    if not rca_details:
        return ""

    lines: list[str] = [
        "## Root Cause Analysis (RCA) — High-Risk DAGs",
        "",
        "Based on recent error logs from `JONATHON_DAG_ERROR_LOG_RECORDS`.",
        "Errors are from past runs of the same DAG.",
        "",
    ]

    for dag_id in sorted(rca_details.keys()):
        detail = rca_details[dag_id]
        s = detail.summary

        lines.append(f"\n### {dag_id}")
        lines.extend([
            f"| Field | Value |",
            f"|---|---|",
            f"| **Error Category** | {s.rca_error_category} |",
            f"| **Top Failing Task** | {s.rca_top_task or '-'} |",
            f"| **Worst Severity** | {s.rca_severity} |",
            f"| **Recurrence** | {s.rca_recurrence} distinct run(s) with errors |",
            f"| **Total Error Records** | {s.rca_error_count} |",
            f"| **Max Retry Number** | {s.rca_try_number_max} |",
            f"| **First Seen** | {s.rca_first_seen or '-'} |",
            f"| **Last Seen** | {s.rca_last_seen or '-'} |",
            f"| **Latest Error** | {s.rca_top_error} |",
        ])

        if s.rca_stack_hint:
            lines.append(f"| **Stack Hint** | {s.rca_stack_hint} |")

        if detail.top_entries:
            lines.extend(["", "**Recent error entries:**", ""])
            for i, entry in enumerate(detail.top_entries, 1):
                lines.append(
                    f"{i}. [{entry['severity']}] **{entry['task_id']}** "
                    f"(try {entry['try_number']}) {entry['log_ts']}"
                )
                lines.append(f"   > {entry['message']}")
                if entry["stack_hint"]:
                    lines.append(f"   > _{entry['stack_hint']}_")

    return "\n".join(lines)
