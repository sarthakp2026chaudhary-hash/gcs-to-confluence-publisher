# config.py

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable


# --- dev_config.json helpers ---

_DEV_CONFIG_PATH = Path(__file__).parent / "dev_config.json"


def load_dev_config() -> dict[str, Any]:
    """Load configuration from dev_config.json located next to this file."""
    with _DEV_CONFIG_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


_dev_cfg_cache: dict[str, Any] | None = None


def _dev_cfg() -> dict[str, Any]:
    """Return a cached copy of dev_config.json."""
    global _dev_cfg_cache
    if _dev_cfg_cache is None:
        _dev_cfg_cache = load_dev_config()
    return _dev_cfg_cache


def table_id(table_name: str) -> str:
    """Return the fully-qualified BigQuery table id ``project.dataset.table``.

    The project and dataset are read from ``dev_config.json``.

    ``table_name`` is resolved through the optional tables mapping so
    individual table names can be overridden without touching source code.

    Example::

        from config import table_id
        full_name = table_id("JONATHON_SOURCE_DAG_RUN")
        # -> "jonathon-gcpbucket-dev.JONATHON_DATASET_DEV.JONATHON_SOURCE_DAG_RUN"
    """
    cfg = _dev_cfg()
    project = cfg["project_id"]
    dataset = cfg["dataset_id"]
    resolved = cfg.get("tables", {}).get(table_name, table_name)
    return f"{project}.{dataset}.{resolved}"


@dataclass(frozen=True)
class PathsConfig:
    artefacts_dir: Path = Path("./artefacts")
    outputs_dir: Path = Path("./outputs")
    output_gcs_uri: str | None = None


@dataclass(frozen=True)
class DBConfig:
    jonathon_db_uri: str
    schema: str | None = None


@dataclass(frozen=True)
class LogsConfig:
    enabled: bool = True
    error_log_table: str | None = None
    max_objects_per_dag_run: int = 200


@dataclass(frozen=True)
class FeatureConfig:
    failure_states: tuple[str, ...] = ("failed", "upstream_failed", "timeout")
    success_states: tuple[str, ...] = ("success",)
    exclude_states: tuple[str, ...] = ("running", "queued", "scheduled")
    windows_days: tuple[int, ...] = (7, 30, 90)
    pressure_window_hours: int = 1
    critical_tasks: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelConfig:
    target_label: str = "label_failed"
    history_days: int = 30
    val_days: int = 14
    test_days: int = 14
    random_state: int = 42
    calibration_method: str = "isotonic"
    high_risk_threshold: float = 0.70
    medium_risk_threshold: float = 0.40
    decision_threshold: float = 0.42
    xgb_n_estimators: int = 350
    xgb_learning_rate: float = 0.05
    xgb_max_depth: int = 5
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.9
    exit_after_diagnostics_logs: bool = False


@dataclass(frozen=True)
class ReportConfig:
    html_enabled: bool = True
    top_n: int = 10


@dataclass(frozen=True)
class BigQueryConfig:
    """BigQuery configuration for accessing tables securely."""
    project_id: str
    dataset_id: str
    # For local dev: use your own GCP SA. For prod: use target SA with impersonation
    target_service_account: str | None = None
    # Optional: for tracking billing/quota to a specific project
    quota_project_id: str | None = None
    # Token lifetime in seconds (for impersonation tokens)
    token_lifetime: int = 3600


@dataclass(frozen=True)
class AppConfig:
    db: DBConfig
    logs: LogsConfig
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    def ensure_dirs(self) -> None:
        self.paths.artefacts_dir.mkdir(parents=True, exist_ok=True)
        self.paths.outputs_dir.mkdir(parents=True, exist_ok=True)


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _as_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple([item.strip() for item in value.split(",") if item.strip()])


def get_prediction_date(date_str: str | None) -> date:
    if date_str:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    return datetime.now(timezone.utc).date()


def load_config() -> AppConfig:
    # Read all application settings from dev_config.json.
    dcfg = _dev_cfg()

    jonathon_db_uri = str(dcfg.get("jonathon_db_uri", "")).strip()
    logs_enabled = bool(dcfg.get("enable_error_log_parsing", True))

    # Build the fully-qualified error log table name from dev_config.json
    error_log_table = table_id("JONATHON_DAG_ERROR_LOG_RECORDS")

    config = AppConfig(
        db=DBConfig(
            jonathon_db_uri=jonathon_db_uri,
            schema=str(dcfg.get("jonathon_db_schema", "")).strip() or None,
        ),
        logs=LogsConfig(
            enabled=logs_enabled,
            error_log_table=error_log_table,
            max_objects_per_dag_run=int(dcfg.get("log_max_objects_per_dag_run", 200)),
        ),
        features=FeatureConfig(
            failure_states=(
                _split_csv(dcfg.get("failure_states"))
                or ("failed", "upstream_failed", "timeout")
            ),
            success_states=(
                _split_csv(dcfg.get("success_states"))
                or ("success",)
            ),
            critical_tasks=_split_csv(dcfg.get("critical_tasks")),
        ),
        model=ModelConfig(
            history_days=int(dcfg.get("history_days", 30)),
            val_days=int(dcfg.get("val_days", 14)),
            test_days=int(dcfg.get("test_days", 14)),
            calibration_method=str(dcfg.get("calibration_method", "isotonic")),
            high_risk_threshold=float(dcfg.get("risk_threshold_high", 0.70)),
            medium_risk_threshold=float(dcfg.get("risk_threshold_med", 0.40)),
            decision_threshold=float(dcfg.get("decision_threshold", 0.42)),
            xgb_n_estimators=int(dcfg.get("xgb_n_estimators", 350)),
            xgb_max_depth=int(dcfg.get("xgb_max_depth", 5)),
            xgb_learning_rate=float(dcfg.get("xgb_learning_rate", 0.05)),
            xgb_subsample=float(dcfg.get("xgb_subsample", 0.9)),
            xgb_colsample_bytree=float(dcfg.get("xgb_colsample", 0.9)),
            exit_after_diagnostics_logs=bool(dcfg.get("exit_after_diagnostics_logs", False)),
        ),
        paths=PathsConfig(
            artefacts_dir=Path(str(dcfg.get("artefacts_dir", "./artefacts"))),
            outputs_dir=Path(str(dcfg.get("outputs_dir", "./outputs"))),
            output_gcs_uri=str(dcfg.get("output_gcs_uri", "")).strip() or None,
        ),
        report=ReportConfig(
            html_enabled=bool(dcfg.get("report_html_enabled", True)),
            top_n=int(dcfg.get("report_top_n", 10)),
        ),
    )

    config.ensure_dirs()
    return config


def date_to_str(dt: date) -> str:
    return dt.strftime("%Y%m%d")


def require_columns(df_columns: Iterable[str], columns: Iterable[str]) -> list[str]:
    col_set = set(df_columns)
    return [column for column in columns if column in col_set]


def load_bq_config() -> BigQueryConfig:
    """Load BigQuery configuration from dev_config.json.

    Optional fields in dev_config.json:
        - target_service_account
        - quota_project_id
        - token_lifetime
    """
    dcfg = _dev_cfg()

    project_id = str(dcfg.get("project_id", "")).strip()
    if not project_id:
        raise ValueError("project_id is not set in dev_config.json.")

    dataset_id = str(dcfg.get("dataset_id", "")).strip()
    if not dataset_id:
        raise ValueError("dataset_id is not set in dev_config.json.")

    target_sa = str(dcfg.get("target_service_account", "")).strip() or None
    quota_project = str(dcfg.get("quota_project_id", "")).strip() or None
    token_lifetime = int(dcfg.get("token_lifetime", 3600))

    return BigQueryConfig(
        project_id=project_id,
        dataset_id=dataset_id,
        target_service_account=target_sa,
        quota_project_id=quota_project or project_id,  # Default to project_id if not set
        token_lifetime=token_lifetime,
    )
