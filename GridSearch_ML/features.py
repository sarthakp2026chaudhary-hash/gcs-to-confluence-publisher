# features.py

from __future__ import annotations

import calendar
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import cast

import numpy as np
import pandas as pd

try:
    from croniter import croniter as _croniter  # type: ignore
    _CRONITER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CRONITER_AVAILABLE = False

from config import FeatureConfig

# ---------------------------------------------------------------------------
# Schedule-interval parsing helpers
# ---------------------------------------------------------------------------

# Preset aliases → canonical 5-field cron expressions
_JONATHON_PRESET_MAP: dict[str, str] = {
    "@hourly":    "0 * * * *",
    "@daily":     "0 0 * * *",
    "@midnight":  "0 0 * * *",
    "@weekly":    "0 0 * * 0",
    "@monthly":   "0 0 1 * *",
    "@quarterly": "0 0 1 */3 *",
    "@yearly":    "0 0 1 1 *",
    "@annually":  "0 0 1 1 *",
}

_NON_RECURRING_PRESETS = {"@once", "@never"}

# Patterns that indicate a non-cron, non-parseable schedule (dataset triggers,
# custom timetables, timedelta objects serialised as JSON, etc.)
_NON_SCHEDULE_RE = re.compile(
    r"dataset|_type|timetable|timedelta",
    re.IGNORECASE,
)


def _is_scheduled_on_date(schedule_interval: object, target_date: date) -> bool | None:
    """Determine whether a DAG fires at least once on *target_date*.

    Returns
    -------
    True   – schedule is parseable and has at least one fire-time on the date.
    False  – schedule is parseable and has *no* fire-time on the date.
    None   – schedule cannot be parsed (null, dataset trigger, custom timetable,
             etc.); caller should treat the DAG as a fallback candidate and
             include it for prediction.
    """
    if schedule_interval is None:
        return None

    try:
        is_na = pd.isna(schedule_interval)  # type: ignore[arg-type]
        if is_na:
            return None
    except (TypeError, ValueError):
        pass

    raw = str(schedule_interval).strip()

    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1].strip()

    if not raw or raw.lower() in ("none", "null", "nan", ""):
        return None

    if raw.lower() in _NON_RECURRING_PRESETS:
        return False

    # Dataset-triggered / timetable objects / timedelta cannot predict
    if _NON_SCHEDULE_RE.search(raw):
        return None

    # Resolve preset aliases
    cron = _JONATHON_PRESET_MAP.get(raw.lower(), raw)

    if not _CRONITER_AVAILABLE:
        logger.debug("croniter not available; treating schedule '%s' as fallback.", raw)
        return None

    try:
        day_start_dt: datetime = datetime(
            target_date.year, target_date.month, target_date.day,
            tzinfo=timezone.utc,
        )
        # Start one second before midnight so schedules firing exactly at 00:00
        # are correctly treated as scheduled for target_date.
        iter_ = _croniter(cron, day_start_dt - timedelta(seconds=1))
        next_fire: datetime = iter_.get_next(datetime)

        day_end_dt = datetime(
            target_date.year, target_date.month, target_date.day, 23, 59, 59,
            tzinfo=timezone.utc,
        )
        return next_fire <= day_end_dt

    except Exception as exc:
        logger.debug("Could not parse schedule_interval '%s': %s", raw, exc)
        return None


logger = logging.getLogger(__name__)

DAY_RUN_REQUIRED_COLUMNS = [
    "dag_id",
    "run_id",
    "logical_date",
    "execution_date",
    "state",
    "start_date",
    "end_date",
    "queued_at",
    "data_interval_start",
    "data_interval_end",
    "run_type",
    "external_trigger",
    "clear_number",
]

DAG_REQUIRED_COLUMNS = [
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
]

TASK_INSTANCE_REQUIRED_COLUMNS = [
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
    "priority_weight",
]


@dataclass
class FeatureArtifacts:
    feature_frame: pd.DataFrame
    dropped_targets: pd.DataFrame


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.mean())


def _safe_std(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.std(ddof=0))


def _safe_quantile(series: pd.Series, q: float) -> float:
    if series.empty:
        return float("nan")
    return float(series.quantile(q))


def _safe_int_flag(value: object) -> int:
    if value is None or pd.isna(value):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _month_end_flag(ts: pd.Timestamp) -> int:
    if pd.isna(ts):
        return 0
    last_day = calendar.monthrange(ts.year, ts.month)[1]
    return int(ts.day >= last_day - 1)


def _compute_label(state: str | None, cfg: FeatureConfig) -> float:
    if state is None or pd.isna(state):
        return np.nan
    normalized = str(state).strip().lower()
    if normalized in {s.lower() for s in cfg.failure_states}:
        return 1.0
    if normalized in {s.lower() for s in cfg.success_states}:
        return 0.0
    return np.nan


def _prepare_dag_runs(dag_run_df: pd.DataFrame) -> pd.DataFrame:
    frame = dag_run_df.copy()
    for col in DAY_RUN_REQUIRED_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA

    if "logical_date" not in frame.columns and "execution_date" in frame.columns:
        frame["logical_date"] = frame["execution_date"]

    datetime_cols = [
        "logical_date",
        "start_date",
        "end_date",
        "queued_at",
        "data_interval_start",
        "data_interval_end",
    ]
    for col in datetime_cols:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], utc=True, errors="coerce")

    frame = frame.sort_values(["dag_id", "logical_date", "run_id"], kind="mergesort").reset_index(drop=True)
    return frame


def _prepare_dag(dag_df: pd.DataFrame) -> pd.DataFrame:
    frame = dag_df.copy()
    for col in DAG_REQUIRED_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA
    if "last_parsed_time" in frame.columns:
        frame["last_parsed_time"] = pd.to_datetime(frame["last_parsed_time"], utc=True, errors="coerce")
    return frame


def _prepare_task_instance(task_instance_df: pd.DataFrame) -> pd.DataFrame:
    frame = task_instance_df.copy()
    for col in TASK_INSTANCE_REQUIRED_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA

    for col in ["start_date", "end_date", "queued_dttm"]:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], utc=True, errors="coerce")

    for col in ["try_number", "max_tries", "duration", "priority_weight"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.sort_values(["dag_id", "run_id", "task_id"], kind="mergesort").reset_index(drop=True)
    return frame


def _build_targets_for_training(
    dag_run_df: pd.DataFrame, cfg: FeatureConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = dag_run_df.copy()
    frame["label_failed"] = frame["state"].apply(lambda value: _compute_label(value, cfg))

    excluded = frame[frame["state"].astype(str).str.lower().isin({s.lower() for s in cfg.exclude_states})].copy()
    trainable = frame[~frame.index.isin(excluded.index)].copy()

    dropped = trainable[trainable["label_failed"].isna()].copy()
    trainable = trainable[~trainable["label_failed"].isna()].copy()

    dropped_targets = pd.concat([excluded, dropped], axis=0, ignore_index=True)
    return cast(pd.DataFrame, trainable), cast(pd.DataFrame, pd.DataFrame(dropped_targets))


def _build_targets_for_date(dag_run_df: pd.DataFrame, target_date: date) -> pd.DataFrame:
    frame = dag_run_df.copy()
    date_start = pd.Timestamp(target_date, tz="UTC")
    date_end = date_start + pd.Timedelta(days=1)

    if "logical_date" not in frame.columns:
        return pd.DataFrame(columns=frame.columns)

    return frame[(frame["logical_date"] >= date_start) & (frame["logical_date"] < date_end)].copy()


def _fallback_targets_from_dag(dag_df: pd.DataFrame, target_date: date) -> pd.DataFrame:
    """Build synthetic target rows for active, non-paused DAGs that had no
    actual run recorded on *target_date*.

    Schedule-interval parsing strategy
    -----------------------------------
    * If schedule_interval is a parseable cron expression (or a preset such as
      @daily) we evaluate whether it fires on *target_date*.
      DAGs whose schedule provably does **not** fire that day are excluded.
    * If schedule_interval is null, a dataset trigger, a custom timetable
      object, or anything else that cannot be parsed, the DAG is **included**
      as a fallback candidate so it still receives a prediction rather than
      being silently dropped.
    """
    logical_date = pd.Timestamp(target_date, tz="UTC")

    if dag_df.empty:
        return pd.DataFrame(columns=["dag_id", "run_id", "logical_date", "state", "run_type", "external_trigger"])

    frame = dag_df.copy()

    is_active = frame["is_active"] if "is_active" in frame.columns else pd.Series(1, index=frame.index)
    is_paused = frame["is_paused"] if "is_paused" in frame.columns else pd.Series(0, index=frame.index)
    active_mask = (is_active.fillna(1).astype(int) == 1) & (is_paused.fillna(0).astype(int) == 0)
    frame = frame[active_mask]

    # Schedule-interval filter
    # For each DAG, determine whether it should have run on target_date:
    #   True  → scheduled       → include
    #   False → not scheduled   → exclude (avoids false-positive predictions)
    #   None  → unparseable (dataset/null/timetable) → include as fallback
    sched_col = "schedule_interval" if "schedule_interval" in frame.columns else None

    if sched_col is not None:
        schedule_decisions: list[bool] = []
        for _, dag_row in frame.iterrows():
            decision = _is_scheduled_on_date(dag_row[sched_col], target_date)
            if decision is False:
                # Provably not scheduled → exclude
                schedule_decisions.append(False)
            else:
                # Scheduled (True) or unparseable/null (None) → include
                schedule_decisions.append(True)

        keep_mask = pd.Series(schedule_decisions, index=frame.index)
        excluded_count = (~keep_mask).sum()
        if excluded_count:
            logger.info(
                "Schedule-interval filter: excluded %d DAG(s) not scheduled on %s.",
                excluded_count,
                target_date,
            )
        frame = frame[keep_mask]

    frame = frame[["dag_id"]].drop_duplicates().copy()
    frame["run_id"] = frame["dag_id"].apply(lambda dag: f"predicted_{target_date.isoformat()}_{dag}")
    frame["logical_date"] = logical_date
    frame["state"] = pd.NA
    frame["run_type"] = "scheduled"
    frame["external_trigger"] = False

    return cast(pd.DataFrame, frame)


def _calculate_feature_rows(
    targets: pd.DataFrame,
    dag_run_df: pd.DataFrame,
    task_instance_df: pd.DataFrame,
    dag_df: pd.DataFrame,
    run_log_features_df: pd.DataFrame,
    cfg: FeatureConfig,
) -> pd.DataFrame:
    targets = targets.copy()

    if "logical_date" in targets.columns:
        targets["logical_date"] = pd.to_datetime(targets["logical_date"], utc=True, errors="coerce")

    targets = targets.sort_values(["dag_id", "logical_date", "run_id"], kind="mergesort").reset_index(drop=True)

    if not run_log_features_df.empty:
        run_log_features_df = run_log_features_df.copy()
        if "logical_date" in run_log_features_df.columns:
            run_log_features_df["logical_date"] = pd.to_datetime(
                run_log_features_df["logical_date"], utc=True, errors="coerce"
            )
        merge_cols = [col for col in ["dag_id", "run_id", "logical_date"] if col in run_log_features_df.columns]
        targets = targets.merge(run_log_features_df, on=merge_cols, how="left", suffixes=("", "_log"))

    dag_map = dag_df.drop_duplicates(subset=["dag_id"], keep="last") if not dag_df.empty else pd.DataFrame()

    results: list[dict[str, object]] = []

    for row in targets.itertuples(index=False):
        dag_id = getattr(row, "dag_id")
        run_id = getattr(row, "run_id")
        logical_date = getattr(row, "logical_date")

        dag_history = dag_run_df[(dag_run_df["dag_id"] == dag_id) & (dag_run_df["logical_date"] < logical_date)].copy()

        task_history = task_instance_df.iloc[0:0].copy()
        if not dag_history.empty:
            dag_history_run_ids = set(dag_history["run_id"].dropna().astype(str).tolist())
            task_history = task_instance_df[
                (task_instance_df["dag_id"] == dag_id)
                & (task_instance_df["run_id"].astype(str).isin(dag_history_run_ids))
            ].copy()

        record: dict[str, object] = {
            "dag_id": dag_id,
            "run_id": run_id,
            "logical_date": logical_date,
            "run_type": getattr(row, "run_type", pd.NA),
            "external_trigger": getattr(row, "external_trigger", pd.NA),
            "clear_number": getattr(row, "clear_number", pd.NA),
        }

        for window in cfg.windows_days:
            start = logical_date - pd.Timedelta(days=window)
            hist = dag_history[dag_history["logical_date"] >= start]
            failed_mask = hist["state"].astype(str).str.lower().isin({s.lower() for s in cfg.failure_states})
            record[f"failure_rate_{window}d"] = float(failed_mask.mean()) if len(hist) else np.nan
            record[f"runs_count_{window}d"] = int(len(hist))

            if window in (30,):
                durations = (hist["end_date"] - hist["start_date"]).dt.total_seconds()
                durations = durations.where(durations.notna(), hist.get("duration", pd.Series(index=hist.index, dtype=float)))
                record["mean_duration_30d"] = _safe_mean(durations.dropna())
                record["p95_duration_30d"] = _safe_quantile(durations.dropna(), 0.95)
                record["duration_std_30d"] = _safe_std(durations.dropna())

        consecutive_failures = 0
        for state in reversed(dag_history["state"].astype(str).str.lower().tolist()):
            if state in {s.lower() for s in cfg.failure_states}:
                consecutive_failures += 1
            else:
                break
        record["consecutive_failures"] = consecutive_failures

        th_30 = task_history[
            task_history["start_date"].fillna(task_history["queued_dttm"]).fillna(pd.Timestamp.min.tz_localize("UTC"))
            >= (logical_date - pd.Timedelta(days=30))
        ]

        task_states = th_30.get("state", pd.Series(dtype=str)).astype(str).str.lower()
        record["avg_retries_30d"] = _safe_mean(th_30.get("try_number", pd.Series(dtype=float)).fillna(0) - 1)
        record["retry_rate_30d"] = float((th_30.get("try_number", 0).fillna(0) > 1).mean()) if len(th_30) else np.nan
        record["failed_tasks_rate_30d"] = float(task_states.isin({"failed", "upstream_failed", "skipped"}).mean()) if len(th_30) else np.nan

        if cfg.critical_tasks:
            critical_mask = th_30["task_id"].astype(str).isin(cfg.critical_tasks)
            critical = th_30[critical_mask]
            critical_state = critical.get("state", pd.Series(dtype=str)).astype(str).str.lower()
            record["critical_tasks_failed_rate_30d"] = (
                float(critical_state.isin({"failed", "upstream_failed"}).mean()) if len(critical) else np.nan
            )
        else:
            record["critical_tasks_failed_rate_30d"] = np.nan

        th_7 = task_history[
            task_history["queued_dttm"].fillna(pd.Timestamp.min.tz_localize("UTC"))
            >= (logical_date - pd.Timedelta(days=7))
        ]

        queue_time_sec = (th_7["start_date"] - th_7["queued_dttm"]).dt.total_seconds()
        record["avg_queue_time_7d"] = _safe_mean(queue_time_sec.dropna())

        running_around_target = dag_run_df[
            (dag_run_df["logical_date"] >= logical_date - pd.Timedelta(hours=cfg.pressure_window_hours))
            & (dag_run_df["logical_date"] <= logical_date + pd.Timedelta(hours=cfg.pressure_window_hours))
            & (dag_run_df["state"].astype(str).str.lower() == "running")
        ]
        record["concurrent_runs_proxy"] = int(len(running_around_target))

        recent_tasks_1h = task_instance_df[
            task_instance_df["queued_dttm"].between(logical_date - pd.Timedelta(hours=1), logical_date, inclusive="both")
        ]
        recent_tasks_24h = task_instance_df[
            task_instance_df["queued_dttm"].between(logical_date - pd.Timedelta(hours=24), logical_date, inclusive="both")
        ]

        record["pool_pressure_proxy_1h"] = int(len(recent_tasks_1h))
        record["pool_pressure_proxy_24h"] = int(len(recent_tasks_24h))

        record["hour_of_day"] = logical_date.hour
        record["day_of_week"] = logical_date.dayofweek
        record["is_weekend"] = int(logical_date.dayofweek >= 5)
        record["month_end_flag"] = _month_end_flag(logical_date)

        if not dag_map.empty and dag_id in set(dag_map["dag_id"]):
            dag_row = dag_map[dag_map["dag_id"] == dag_id].iloc[-1]
            record["dag_last_parsed_age_hours"] = (
                (logical_date - dag_row.get("last_parsed_time")).total_seconds() / 3600
                if pd.notna(dag_row.get("last_parsed_time"))
                else np.nan
            )
            record["dag_paused_flag"] = int(bool(dag_row.get("is_paused", 0)))
            record["dag_is_active_flag"] = int(bool(dag_row.get("is_active", 1)))
            record["has_import_errors"] = int(bool(dag_row.get("has_import_errors", 0)))
            record["max_active_runs"] = pd.to_numeric(dag_row.get("max_active_runs", np.nan), errors="coerce")
            record["max_active_tasks"] = pd.to_numeric(dag_row.get("max_active_tasks", np.nan), errors="coerce")
            record["max_consecutive_failed_dag_runs"] = pd.to_numeric(
                dag_row.get("max_consecutive_failed_dag_runs", np.nan), errors="coerce"
            )
            record["schedule_interval"] = dag_row.get("schedule_interval", pd.NA)
            record["owners"] = dag_row.get("owners", pd.NA)
            record["fileloc"] = dag_row.get("fileloc", pd.NA)
        else:
            record.update({
                "dag_last_parsed_age_hours": np.nan,
                "dag_paused_flag": np.nan,
                "dag_is_active_flag": np.nan,
                "has_import_errors": np.nan,
                "max_active_runs": np.nan,
                "max_active_tasks": np.nan,
                "max_consecutive_failed_dag_runs": np.nan,
                "schedule_interval": pd.NA,
                "owners": pd.NA,
                "fileloc": pd.NA,
            })

        if "stacktrace_seen_rate" in targets.columns:
            record["stacktrace_seen_rate_30d"] = getattr(row, "stacktrace_seen_rate", np.nan)
        else:
            record["stacktrace_seen_rate_30d"] = np.nan

        for column_name in [
            "error_cat_permission_auth",
            "error_cat_quota_rate_limit",
            "error_cat_not_found",
            "error_cat_timeout",
            "error_cat_oom_memory",
            "error_cat_network_connection",
            "error_cat_other",
            "kw_retrying",
            "kw_killed",
            "kw_sigterm",
            "kw_broken_dag",
            "kw_import_error",
        ]:
            record[f"{column_name}_30d"] = getattr(row, column_name, np.nan)

        if "label_failed" in targets.columns:
            record["label_failed"] = getattr(row, "label_failed")

        if "state" in targets.columns:
            record["state"] = getattr(row, "state")

        results.append(record)

    return pd.DataFrame(results)


def data_quality_checks(frame: pd.DataFrame, label_col: str = "label_failed") -> dict[str, object]:
    checks: dict[str, object] = {
        "rows": int(len(frame)),
        "duplicate_keys": int(frame.duplicated(subset=["dag_id", "run_id"], keep=False).sum()) if not frame.empty else 0,
        "null_rate_by_column": frame.isna().mean().sort_values(ascending=False).head(10).to_dict() if not frame.empty else {},
    }
    if label_col in frame.columns and not frame.empty:
        checks["label_distribution"] = frame[label_col].value_counts(dropna=False).to_dict()
    return checks


def build_training_features(
    dag_run_df: pd.DataFrame,
    task_instance_df: pd.DataFrame,
    dag_df: pd.DataFrame,
    run_log_features_df: pd.DataFrame,
    cfg: FeatureConfig,
) -> FeatureArtifacts:
    dag_run = _prepare_dag_runs(dag_run_df)
    task_instance = _prepare_task_instance(task_instance_df)
    dag = _prepare_dag(dag_df)

    train_targets, dropped = _build_targets_for_training(dag_run, cfg)

    feature_frame = _calculate_feature_rows(
        targets=train_targets,
        dag_run_df=dag_run,
        task_instance_df=task_instance,
        dag_df=dag,
        run_log_features_df=run_log_features_df,
        cfg=cfg,
    )

    logger.info("Built training feature frame. rows=%s cols=%s", len(feature_frame), len(feature_frame.columns))
    return FeatureArtifacts(feature_frame=feature_frame, dropped_targets=dropped)


def build_inference_features(
    target_date: date,
    dag_run_df: pd.DataFrame,
    task_instance_df: pd.DataFrame,
    dag_df: pd.DataFrame,
    run_log_features_df: pd.DataFrame,
    cfg: FeatureConfig,
) -> FeatureArtifacts:
    dag_run = _prepare_dag_runs(dag_run_df)
    task_instance = _prepare_task_instance(task_instance_df)
    dag = _prepare_dag(dag_df)

    targets = _build_targets_for_date(dag_run, target_date)
    fallback_targets = _fallback_targets_from_dag(dag, target_date)

    if targets.empty:
        logger.warning(
            "No dag_run rows found for target date %s. Falling back to active DAGs next-run approximation.",
            target_date,
        )
        targets = fallback_targets
    else:
        existing_dag_ids = set(targets["dag_id"].dropna().astype(str).tolist()) if "dag_id" in targets.columns else set()
        missing_active = fallback_targets[~fallback_targets["dag_id"].astype(str).isin(existing_dag_ids)].copy()
        if not missing_active.empty:
            logger.info(
                "Inference coverage expansion: adding %s active DAG(s) without a target-date run.",
                len(missing_active),
            )
            targets = pd.concat([targets, missing_active], ignore_index=True, sort=False)

    feature_frame = _calculate_feature_rows(
        targets=targets,
        dag_run_df=dag_run,
        task_instance_df=task_instance,
        dag_df=dag,
        run_log_features_df=run_log_features_df,
        cfg=cfg,
    )

    logger.info("Built inference feature frame for %s. rows=%s", target_date, len(feature_frame))
    return FeatureArtifacts(feature_frame=feature_frame, dropped_targets=pd.DataFrame())
