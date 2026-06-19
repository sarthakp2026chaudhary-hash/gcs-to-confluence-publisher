# rca_failed_dags_report.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

from config import AppConfig, date_to_str
from rca import _classify_error, _extract_stack_hint
from storage_utils import GCSOutputStore

logger = logging.getLogger(__name__)


@dataclass
class FailedRunsRCAReportOutputs:
    csv_path: Path
    markdown_path: Path
    html_path: Path | None
    rows: int


def _build_failed_runs_query(table_name: str, start_ts: datetime, end_ts: datetime) -> str:
    return f"""
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
FROM `{table_name}`
WHERE log_ts > TIMESTAMP('{start_ts.isoformat()}')
  AND log_ts < TIMESTAMP('{end_ts.isoformat()}')
  AND UPPER(COALESCE(severity, '')) IN ('ERROR', 'CRITICAL')
ORDER BY log_ts DESC
LIMIT 200000
""".strip()


def _summarize_failed_runs(log_rows: pd.DataFrame) -> pd.DataFrame:
    if log_rows.empty:
        return pd.DataFrame(
            columns=[
                "dag_id",
                "run_id",
                "log_count",
                "top_failing_task",
                "max_try_number",
                "worst_severity",
                "latest_log_ts",
                "latest_message",
                "error_category",
                "stack_hint",
            ]
        )

    rows: list[dict[str, object]] = []
    severity_rank = {"CRITICAL": 4, "ERROR": 3, "WARNING": 2, "INFO": 1}

    grouped = log_rows.groupby(["dag_id", "run_id"], dropna=False)

    for key, group in grouped:
        if isinstance(key, tuple) and len(key) == 2:
            dag_id, run_id = key
        else:
            dag_id = key
            run_id = ""

        group_sorted = group.sort_values("log_ts", ascending=False)
        top_row = group_sorted.iloc[0]

        task_counts = group["task_id"].dropna().astype(str).value_counts()
        top_task = str(task_counts.index[0]) if not task_counts.empty else ""

        severities = group["severity"].dropna().astype(str).str.upper().tolist()
        worst_sev = "INFO"
        worst_rank = 0
        for sev in severities:
            rank = severity_rank.get(sev, 0)
            if rank > worst_rank:
                worst_rank = rank
                worst_sev = sev

        latest_message = str(top_row.get("message", ""))[:300]
        full_block = str(top_row.get("full_block", "")) if pd.notna(top_row.get("full_block")) else ""
        combined = f"{latest_message} {full_block}"

        rows.append({
            "dag_id": str(dag_id),
            "run_id": str(run_id),
            "log_count": int(len(group)),
            "top_failing_task": top_task,
            "max_try_number": int(group["try_number"].max()) if not group["try_number"].isna().all() else 0,
            "worst_severity": worst_sev,
            "latest_log_ts": str(top_row.get("log_ts", ""))[:19],
            "latest_message": latest_message,
            "error_category": _classify_error(combined),
            "stack_hint": _extract_stack_hint(full_block),
        })

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(
        by=["worst_severity", "log_count", "latest_log_ts"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    return summary_df


def _render_markdown(summary_df: pd.DataFrame, target_date: date, lookback_days: int) -> str:
    lines = [
        f"# Failed DAG Runs RCA Report — {target_date.isoformat()}",
        "",
        "## Scope",
        "",
        f"- Lookback window: last {lookback_days} day(s)",
        f"- Failed dag-run entries found: {len(summary_df)}",
        f"- Source: `JONATHON_DAG_ERROR_LOG_RECORDS` (severity ERROR/CRITICAL)",
        "",
        "## Failed DAG Runs with RCA",
        "",
    ]

    if summary_df.empty:
        lines.append("No failed dag-run records found in the lookback window.")
        return "\n".join(lines)

    lines.extend([
        "| dag_id | run_id | severity | log_count | top_task | max_try | category | latest_message |",
        "|---|---|---|---|---|---|---|---|",
    ])

    for row in summary_df.itertuples(index=False):
        lines.append(
            "| " + " | ".join([
                str(getattr(row, "dag_id", "")),
                str(getattr(row, "run_id", "")),
                str(getattr(row, "worst_severity", "")),
                str(getattr(row, "log_count", 0)),
                str(getattr(row, "top_failing_task", "")),
                str(getattr(row, "max_try_number", 0)),
                str(getattr(row, "error_category", "")),
                str(getattr(row, "latest_message", "")).replace("|", ""),
            ]) + " |"
        )

    return "\n".join(lines)


def _markdown_to_simple_html(markdown_text: str) -> str:
    escaped = markdown_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = "<html><body><pre style='font-family:Segoe UI, Arial, sans-serif; white-space: pre-wrap;'>"
    html += escaped
    html += "</pre></body></html>"
    return html


def generate_failed_dags_rca_report(
    config: AppConfig,
    target_date: date,
    lookback_days: int = 7,
) -> FailedRunsRCAReportOutputs:
    if not config.logs.error_log_table:
        raise ValueError("JONATHON_ERROR_LOG_TABLE must be configured for failed-runs RCA report.")

    config.ensure_dirs()

    end_ts = datetime.combine(target_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    start_ts = end_ts - timedelta(days=lookback_days)

    client = bigquery.Client()
    sql = _build_failed_runs_query(config.logs.error_log_table, start_ts=start_ts, end_ts=end_ts)

    logger.info("Running failed-runs RCA query against %s", config.logs.error_log_table)
    log_rows = client.query(sql).to_dataframe()

    if "log_ts" in log_rows.columns:
        log_rows["log_ts"] = pd.to_datetime(log_rows["log_ts"], utc=True, errors="coerce")

    summary_df = _summarize_failed_runs(log_rows)

    date_tag = date_to_str(target_date)

    csv_path = config.paths.outputs_dir / f"failed_runs_rca_{date_tag}.csv"
    md_path = config.paths.outputs_dir / f"failed_runs_rca_report_{date_tag}.md"

    summary_df.to_csv(csv_path, index=False)

    markdown = _render_markdown(summary_df, target_date=target_date, lookback_days=lookback_days)
    md_path.write_text(markdown, encoding="utf-8")

    html_path: Path | None = None
    if config.report.html_enabled:
        html_path = config.paths.outputs_dir / f"failed_runs_rca_report_{date_tag}.html"
        html_path.write_text(_markdown_to_simple_html(markdown), encoding="utf-8")

    if config.paths.output_gcs_uri:
        gcs_store = GCSOutputStore(config.paths.output_gcs_uri)
        gcs_store.upload_file(csv_path, remote_name=f"outputs/{csv_path.name}")
        gcs_store.upload_file(md_path, remote_name=f"outputs/{md_path.name}")
        if html_path is not None:
            gcs_store.upload_file(html_path, remote_name=f"outputs/{html_path.name}")

    logger.info("Failed-runs RCA report generated: %s rows", len(summary_df))

    return FailedRunsRCAReportOutputs(
        csv_path=csv_path,
        markdown_path=md_path,
        html_path=html_path,
        rows=len(summary_df),
    )
