# report.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from config import AppConfig, date_to_str
from rca import build_rca_details_for_report, render_rca_markdown
from storage_utils import GCSOutputStore

logger = logging.getLogger(__name__)


@dataclass
class ReportOutputs:
    markdown_path: Path
    html_path: Path | None


def _load_predictions(config: AppConfig, target_date: date) -> pd.DataFrame:
    path = config.paths.outputs_dir / f"grid_cv_predictions_{date_to_str(target_date)}.csv"

    if path.exists():
        df = pd.read_csv(path)
    elif config.paths.output_gcs_uri:
        gcs_store = GCSOutputStore(config.paths.output_gcs_uri)
        remote_name = f"outputs/{path.name}"
        logger.info("Local predictions file missing; attempting GCS fallback: %s", remote_name)
        df = gcs_store.download_csv(remote_name)
    else:
        raise FileNotFoundError(f"Predictions CSV not found: {path}")

    if "logical_date" in df.columns:
        df["logical_date"] = pd.to_datetime(df["logical_date"], utc=True, errors="coerce")

    return df


def _load_previous_day(config: AppConfig, target_date: date) -> pd.DataFrame:
    prev = target_date - timedelta(days=1)
    path = config.paths.outputs_dir / f"grid_cv_predictions_{date_to_str(prev)}.csv"

    if path.exists():
        return pd.read_csv(path)

    if config.paths.output_gcs_uri:
        gcs_store = GCSOutputStore(config.paths.output_gcs_uri)
        remote_name = f"outputs/{path.name}"
        logger.info("Previous-day local predictions file missing; attempting GCS fallback: %s", remote_name)
        try:
            return gcs_store.download_csv(remote_name)
        except FileNotFoundError:
            logger.info("Previous-day predictions not found in GCS: %s", remote_name)
            return pd.DataFrame()

    return pd.DataFrame()


def _render_markdown(
    df: pd.DataFrame,
    prev_df: pd.DataFrame,
    target_date: date,
    top_n: int,
    rca_markdown: str = "",
) -> str:
    total_runs = len(df)
    high_risk = int((df["risk_bucket"] == "High").sum()) if "risk_bucket" in df.columns else 0

    sorted_df = df.sort_values("predicted_failure_probability", ascending=False)

    # Always include ALL High-risk rows; pad with Med/Low up to top_n if slots remain
    high_rows = sorted_df[sorted_df["risk_bucket"] == "High"] if "risk_bucket" in sorted_df.columns else pd.DataFrame()
    other_rows = sorted_df[sorted_df["risk_bucket"] != "High"] if "risk_bucket" in sorted_df.columns else sorted_df
    remaining_slots = max(0, top_n - len(high_rows))
    top = pd.concat([high_rows, other_rows.head(remaining_slots)], ignore_index=True)

    if not prev_df.empty and "risk_bucket" in prev_df.columns:
        prev_high = int((prev_df["risk_bucket"] == "High").sum())
        trend = high_risk - prev_high
        trend_line = f"- High-risk count change vs yesterday: {trend:+d} ({prev_high} -> {high_risk})"
    else:
        trend_line = "- Trend vs yesterday unavailable (no previous predictions file)."

    lines = [
        f"# DAG Failure Risk Report — {target_date.isoformat()}",
        "",
        "## Summary",
        "",
        f"- Total runs scored: {total_runs}",
        f"- High-risk runs: {high_risk}",
        trend_line,
        "",
        "## Top High-Risk DAG Runs",
        "",
    ]

    if top.empty:
        lines.append("No predictions available.")
    else:
        lines.extend([
            "| dag_id | logical_date | probability | bucket | top_drivers | suggested_action |",
            "|---|---|---|---|---|---|",
        ])

        for row in top.itertuples(index=False):
            lines.append(
                "| " + " | ".join([
                    str(getattr(row, "dag_id", "")),
                    str(getattr(row, "logical_date", "")),
                    f"{float(getattr(row, 'predicted_failure_probability', 0.0)):.3f}",
                    str(getattr(row, "risk_bucket", "")),
                    str(getattr(row, "top_drivers", "")),
                    str(getattr(row, "suggested_action", "")),
                ]) + " |"
            )

    # Append RCA section rendered by rca.py
    if rca_markdown:
        lines.append("")
        lines.append(rca_markdown)

    return "\n".join(lines)


def _markdown_to_simple_html(markdown_text: str) -> str:
    escaped = markdown_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = "<html><body><pre style='font-family:Segoe UI, Arial, sans-serif; white-space: pre-wrap;'>"
    html += escaped
    html += "</pre></body></html>"
    return html


def generate_report(config: AppConfig, target_date: date) -> ReportOutputs:
    config.ensure_dirs()

    df = _load_predictions(config, target_date)
    prev_df = _load_previous_day(config, target_date)

    # Load RCA details and render markdown section via rca.py
    rca_details = build_rca_details_for_report(df, config.logs)
    rca_markdown = render_rca_markdown(rca_details)

    markdown = _render_markdown(df, prev_df, target_date, config.report.top_n, rca_markdown)

    md_path = config.paths.outputs_dir / f"report_{date_to_str(target_date)}.md"
    md_path.write_text(markdown, encoding="utf-8")
    logger.info("Markdown report written: %s", md_path)

    html_path: Path | None = None
    if config.report.html_enabled:
        html_path = config.paths.outputs_dir / f"report_{date_to_str(target_date)}.html"
        html_path.write_text(_markdown_to_simple_html(markdown), encoding="utf-8")
        logger.info("HTML report written: %s", html_path)

    if config.paths.output_gcs_uri:
        gcs_store = GCSOutputStore(config.paths.output_gcs_uri)
        gcs_store.upload_file(md_path, remote_name=f"outputs/{md_path.name}")
        if html_path is not None:
            gcs_store.upload_file(html_path, remote_name=f"outputs/{html_path.name}")

    return ReportOutputs(markdown_path=md_path, html_path=html_path)
