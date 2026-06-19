# extract_db.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, cast

import pandas as pd
from google.cloud import bigquery

from config import AppConfig, table_id

logger = logging.getLogger(__name__)

# Deduplication config per table: (partition_key_columns, insert_time_column_name)
# ROW_NUMBER() keeps only the most recent insert
# Tables are loaded hourly; QUALIFY ROW
TABLE_DEDUP_CONFIG: dict[str, tuple[list[str], str]] = {
    "JONATHON_SOURCE_DAG_RUN": (["dag_id", "run_id"], "insert_time"),
    "JONATHON_SOURCE_TASK_INSTANCE": (["dag_id", "run_id", "task_id", "map_index"], "insert_time"),
    "JONATHON_SOURCE_DAG": (["dag_id"], "insert_time"),
}


def _build_full_names() -> dict[str, str]:
    """Build the fully-qualified BigQuery table name map from dev_config.json.

    Using ``table_id()`` means project/dataset only need to change in one place
    (dev_config.json) and all table references update automatically.
    """
    table_keys = [
        "JONATHON_SOURCE_DAG_RUN",
        "JONATHON_SOURCE_TASK_INSTANCE",
        "JONATHON_SOURCE_DAG",
    ]
    return {key: table_id(key) for key in table_keys}


# Populated once at import time so look-ups stay O(1) throughout the module.
TABLE_FULL_NAMES: dict[str, str] = _build_full_names()


SQL_DAG_RUN_BASE = """
SELECT
    dag_id,
    run_id,
    execution_date,
    state,
    start_date,
    end_date,
    run_type,
    external_trigger,
    queued_at,
    data_interval_start,
    data_interval_end,
    updated_at,
    clear_number
FROM JONATHON_SOURCE_DAG_RUN
""".strip()

SQL_TASK_INSTANCE_BASE = """
SELECT
    dag_id,
    task_id,
    run_id,
    state,
    try_number,
    max_tries,
    start_date,
    end_date,
    duration,
    queued_dttm,
    pool,
    queue,
    priority_weight,
    hostname,
    operator,
    custom_operator_name
FROM JONATHON_SOURCE_TASK_INSTANCE
""".strip()

SQL_DAG_BASE = """
SELECT
    dag_id,
    is_paused,
    is_active,
    last_parsed_time,
    schedule_interval,
    fileloc,
    owners,
    has_import_errors,
    max_active_runs,
    max_active_tasks,
    max_consecutive_failed_dag_runs
FROM JONATHON_SOURCE_DAG
""".strip()


TABLE_COLUMNS: dict[str, list[str]] = {
    "JONATHON_SOURCE_DAG_RUN": [
        "dag_id",
        "run_id",
        "execution_date",
        "state",
        "start_date",
        "end_date",
        "run_type",
        "external_trigger",
        "queued_at",
        "data_interval_start",
        "data_interval_end",
        "updated_at",
        "clear_number",
    ],
    "JONATHON_SOURCE_TASK_INSTANCE": [
        "dag_id",
        "task_id",
        "run_id",
        "state",
        "try_number",
        "max_tries",
        "start_date",
        "end_date",
        "duration",
        "queued_dttm",
        "pool",
        "queue",
        "priority_weight",
        "hostname",
        "operator",
        "custom_operator_name",
    ],
    "JONATHON_SOURCE_DAG": [
        "dag_id",
        "is_paused",
        "is_active",
        "last_parsed_time",
        "schedule_interval",
        "fileloc",
        "owners",
        "has_import_errors",
        "max_active_runs",
        "max_active_tasks",
        "max_consecutive_failed_dag_runs",
    ],
}


@dataclass
class ExtractedData:
    dag_run: pd.DataFrame
    task_instance: pd.DataFrame
    dag: pd.DataFrame


class JonathonMetadataExtractor:
    """Extracts pipeline metadata from BigQuery tables into pandas DataFrames."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.bq_client: bigquery.Client | None = None
        self.bq_client = bigquery.Client()
        logger.info("Initialized BigQuery extractor")

    @staticmethod
    def _format_bigquery_literal(value: Any) -> str:
        if value is None:
            return "NULL"
        if isinstance(value, pd.Timestamp):
            timestamp = value.to_pydatetime()
            if timestamp.tzinfo is None:
                return f"TIMESTAMP('{timestamp.isoformat(sep=' ')}')"
            return f"TIMESTAMP('{timestamp.isoformat()}')"
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return f"TIMESTAMP('{value.isoformat(sep=' ')}')"
            return f"TIMESTAMP('{value.isoformat()}')"
        if isinstance(value, date):
            return f"DATE '{value.isoformat()}'"
        if isinstance(value, str):
            return "'" + value.replace("'", "''") + "'"
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        return str(value)

    def _render_bigquery_where(self, where_clause: str | None, bind_params: dict[str, Any] | None = None) -> str | None:
        if not where_clause:
            return None
        rendered = where_clause
        for key, value in (bind_params or {}).items():
            rendered = rendered.replace(f":{key}", self._format_bigquery_literal(value))
        return rendered

    def _build_in_clause(self, column_name: str, values: list[Any], chunk_size: int = 1000) -> str | None:
        cleaned_values: list[Any] = []
        for value in values:
            if pd.isna(value):
                continue
            cleaned_values.append(value)

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

    @staticmethod
    def _combine_where_clauses(*clauses: str | None) -> str | None:
        valid_clauses = [clause for clause in clauses if clause]
        if not valid_clauses:
            return None
        return " AND ".join(valid_clauses)

    def _select_table(
        self,
        table_name: str,
        required_cols: list[str],
        where_clause: str | None = None,
        bind_params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        full_table = TABLE_FULL_NAMES.get(table_name)
        if not full_table:
            logger.warning("No full table mapping configured for %s. Returning empty.", table_name)
            return pd.DataFrame(columns=required_cols)

        assert self.bq_client is not None

        # Build dedup clause using QUALIFY so only the most recent insert per key is kept
        dedup_conf = TABLE_DEDUP_CONFIG.get(table_name)
        if dedup_conf:
            partition_cols, insert_col = dedup_conf
            partition_expr = ", ".join(f"`{col}`" for col in partition_cols)
            qualify_clause = f"QUALIFY ROW_NUMBER() OVER (PARTITION BY {partition_expr} ORDER BY {insert_col} DESC) = 1"
        else:
            qualify_clause = None

        rendered_where = self._render_bigquery_where(where_clause, bind_params)

        sql_parts = [f"SELECT * FROM `{full_table}`"]
        if rendered_where:
            sql_parts.append(f"WHERE {rendered_where}")
        if qualify_clause:
            sql_parts.append(qualify_clause)

        sql = " ".join(sql_parts)

        try:
            logger.info("Running BigQuery query for table %s", full_table)
            frame = self.bq_client.query(sql).to_dataframe()
        except Exception as exc:
            logger.warning("Error loading table %s: %s", full_table, exc)
            return pd.DataFrame(columns=required_cols)

        present = [c for c in required_cols if c in frame.columns]
        missing = sorted(set(required_cols) - set(present))
        if missing:
            logger.warning("Table %s missing columns: %s", table_name, missing)
        for col in required_cols:
            if col not in frame.columns:
                frame[col] = pd.NA

        # Python-side safety dedup in case BigQuery QUALIFY didn't fully dedup
        if dedup_conf:
            partition_cols, insert_col = dedup_conf
            dedup_cols = [col for col in partition_cols if col in frame.columns]
            if dedup_cols:
                insert_col_present = insert_col if insert_col in frame.columns else None
                if insert_col_present:
                    frame = (
                        frame.sort_values(insert_col_present, ascending=False)
                        .drop_duplicates(subset=dedup_cols, keep="first")
                        .reset_index(drop=True)
                    )
                else:
                    frame = frame.drop_duplicates(subset=dedup_cols, keep="first")

        result = frame.loc[:, required_cols].copy()
        return cast(pd.DataFrame, result)

    @staticmethod
    def _normalize_datetimes(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        for column in columns:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
        return frame

    def fetch(self, start_date: date | None = None, end_date: date | None = None) -> ExtractedData:
        logger.info("Starting metadata extraction. start_date=%s end_date=%s", start_date, end_date)

        where_parts: list[str] = []
        params: dict[str, Any] = {}

        if start_date is not None:
            where_parts.append("execution_date >= :start_date")
            params["start_date"] = pd.Timestamp(start_date)

        if end_date is not None:
            where_parts.append("execution_date < :end_date")
            params["end_date"] = pd.Timestamp(end_date) + pd.Timedelta(days=1)

        dag_run_where = " AND ".join(where_parts) if where_parts else None

        dag_run = self._select_table(
            "JONATHON_SOURCE_DAG_RUN",
            TABLE_COLUMNS["JONATHON_SOURCE_DAG_RUN"],
            where_clause=dag_run_where,
            bind_params=params,
        )

        relevant_dag_ids = dag_run["dag_id"].dropna().astype(str).drop_duplicates().tolist()
        dag_id_filter = self._build_in_clause("dag_id", relevant_dag_ids)

        task_instance_where = dag_id_filter

        task_instance = self._select_table(
            "JONATHON_SOURCE_TASK_INSTANCE",
            TABLE_COLUMNS["JONATHON_SOURCE_TASK_INSTANCE"],
            where_clause=task_instance_where,
        )

        if relevant_dag_ids:
            dag = self._select_table(
                "JONATHON_SOURCE_DAG",
                TABLE_COLUMNS["JONATHON_SOURCE_DAG"],
                where_clause=dag_id_filter,
            )
        else:
            dag = self._select_table("JONATHON_SOURCE_DAG", TABLE_COLUMNS["JONATHON_SOURCE_DAG"])

        dag_run = self._normalize_datetimes(
            dag_run,
            [
                "execution_date",
                "start_date",
                "end_date",
                "queued_at",
                "data_interval_start",
                "data_interval_end",
                "updated_at",
            ],
        )
        task_instance = self._normalize_datetimes(task_instance, ["start_date", "end_date", "queued_dttm"])
        dag = self._normalize_datetimes(dag, ["last_parsed_time"])

        if "execution_date" in dag_run.columns:
            dag_run["logical_date"] = dag_run["execution_date"]

        logger.info(
            "Extraction complete. dag_run=%s task_instance=%s dag=%s",
            len(dag_run),
            len(task_instance),
            len(dag),
        )

        return ExtractedData(
            dag_run=dag_run,
            task_instance=task_instance,
            dag=dag,
        )
