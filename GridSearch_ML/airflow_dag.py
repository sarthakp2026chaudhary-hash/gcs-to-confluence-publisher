# jonathon_dag.py

"""DAG definitions for daily DAG failure risk ML pipeline."""

from airflow.models import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path

default_args = {
    "owner": "data-eng",
    "depends_on_past": False,
    "start_date": datetime(2026, 5, 6),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag_train_only = DAG(
    dag_id="jonathon_dag_failure_risk_grid_cv_ml_train_only",
    default_args=default_args,
    schedule_interval=None,  # Manual trigger only (development)
    catchup=False,
    doc_md="""
# DAG Failure Risk — Train Only

Runs training on historical data (lookback from HISTORY_DAYS, default 30).

**Configuration source:**
- `ML/dev_config.json`

**Outputs:**
- `artefacts/model_bundle.joblib`
- metrics JSON and feature frame artifacts
""",
)

dag_predict_only = DAG(
    dag_id="jonathon_dag_failure_risk_grid_cv_ml_predict_only",
    default_args=default_args,
    schedule_interval=None,  # Manual trigger only (development)
    catchup=False,
    doc_md="""
# DAG Failure Risk — Predict Only

Scores today's scheduled DAG runs using an existing trained model
and generates Markdown/HTML reports.

**Configuration source:**
- `ML/dev_config.json`

**Outputs:**
- `predictions_YYYYMMDD.csv`
- `report_YYYYMMDD.md/.html`
""",
)

dag_failed_rca = DAG(
    dag_id="jonathon_dag_failed_runs_grid_cv_rca_report",
    default_args=default_args,
    schedule_interval=None,  # Manual trigger only (development)
    catchup=False,
    doc_md="""
# DAG Failed Runs RCA Report

Generates RCA report for previously failed DAG runs from
`JONATHON_DAG_ERROR_LOG_RECORDS` (ERROR/CRITICAL severities).

**Outputs:**
- `failed_runs_rca_YYYYMMDD.csv`
- `failed_runs_rca_report_YYYYMMDD.md/.html`
""",
)

logger = logging.getLogger(__name__)


def check_dependencies(**context):
    """Fail fast if required Python dependencies are missing in runtime."""
    required_modules = {
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "xgboost": "xgboost",
        "google.cloud.bigquery": "google-cloud-bigquery",
        "joblib": "joblib",
        "pyarrow": "pyarrow",
    }
    optional_modules = {
        "shap": "shap",
    }

    missing_required: list[str] = []
    missing_optional: list[str] = []

    for module_name, package_name in required_modules.items():
        try:
            __import__(module_name)
        except Exception:
            missing_required.append(package_name)

    for module_name, package_name in optional_modules.items():
        try:
            __import__(module_name)
        except Exception:
            missing_optional.append(package_name)

    if missing_optional:
        logger.warning("Optional packages missing: %s", ", ".join(missing_optional))

    if missing_required:
        raise RuntimeError(
            "Missing required Python packages in runtime: "
            f"{', '.join(missing_required)}. "
            "Install from requirements.txt in your environment."
        )

    logger.info("Dependency check passed: required runtime libraries are available.")
    return {
        "dependency_check": "passed",
        "missing_optional": missing_optional,
    }


def run_pipeline_all(**context):
    """Run the complete ML pipeline: train + predict + report for the execution date.

    This is the main task that orchestrates the entire workflow.
    """
    execution_date = context["execution_date"].date()
    logger.info(f"Starting ML pipeline for date: {execution_date}")

    # Add ML predict module to path
    ml_predict_path = Path(__file__).parent
    if str(ml_predict_path) not in sys.path:
        sys.path.insert(0, str(ml_predict_path))

    # Import pipeline modules
    from config import load_config, date_to_str
    from train import EarlyExitAfterDiagnosticsLogs, train_pipeline
    from predict import predict_for_date
    from report import generate_report

    try:
        config = load_config()
        logger.info("Configuration loaded successfully")

        history_days = config.model.history_days
        history_start = execution_date - timedelta(days=history_days)

        # 1. TRAIN
        logger.info(f"Training on data from {history_start} to {execution_date}")
        try:
            train_outputs = train_pipeline(config, start_date=history_start, end_date=execution_date)
        except EarlyExitAfterDiagnosticsLogs as exc:
            logger.info("Pipeline intentionally stopped after diagnostics logs: %s", exc)
            return {
                "status": "stopped_after_diagnostics_logs",
                "execution_date": execution_date.isoformat(),
            }

        logger.info(
            f"Training complete. Metrics saved: {train_outputs.metrics_path}. "
            f"Model saved: {train_outputs.model_path}"
        )

        # 2. PREDICT
        logger.info(f"Scoring DAG runs for {execution_date}")
        pred_outputs = predict_for_date(config, target_date=execution_date)
        logger.info(f"Predictions saved: {pred_outputs.output_csv} ({len(pred_outputs.predictions)} runs)")

        # 3. REPORT
        logger.info(f"Generating report for {execution_date}")
        report_outputs = generate_report(config, target_date=execution_date)
        logger.info(
            f"Reports saved. Markdown: {report_outputs.markdown_path}. "
            f"HTML: {report_outputs.html_path}"
        )

        # Log summary stats
        high_risk = (pred_outputs.predictions["risk_bucket"] == "High").sum()
        med_risk = (pred_outputs.predictions["risk_bucket"] == "Med").sum()
        logger.info(
            f"Risk summary: {len(pred_outputs.predictions)} total runs, "
            f"{high_risk} High-risk, {med_risk} Med-risk"
        )

        return {
            "predictions_csv": str(pred_outputs.output_csv),
            "report_markdown": str(report_outputs.markdown_path),
            "report_html": str(report_outputs.html_path) if report_outputs.html_path else None,
            "model_path": str(train_outputs.model_path),
            "execution_date": execution_date.isoformat(),
        }

    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}", exc_info=True)
        raise


def run_pipeline_predict_only(**context):
    """Run prediction only (assumes model already trained).

    Useful for fast intraday re-scoring without retraining.
    """
    execution_date = context["execution_date"].date()
    logger.info(f"Running predict-only for date: {execution_date}")

    ml_predict_path = Path(__file__).parent
    if str(ml_predict_path) not in sys.path:
        sys.path.insert(0, str(ml_predict_path))

    from config import load_config
    from predict import predict_for_date
    from report import generate_report

    try:
        config = load_config()
        logger.info("Configuration loaded successfully")

        # Predict
        logger.info(f"Scoring DAG runs for {execution_date}")
        pred_outputs = predict_for_date(config, target_date=execution_date)
        logger.info(f"Predictions saved: {pred_outputs.output_csv}")

        # Report
        logger.info(f"Generating report for {execution_date}")
        report_outputs = generate_report(config, target_date=execution_date)
        logger.info(f"Report saved: {report_outputs.markdown_path}")

        return {
            "predictions_csv": str(pred_outputs.output_csv),
            "report_markdown": str(report_outputs.markdown_path),
        }

    except Exception as exc:
        logger.error(f"Predict task failed: {exc}", exc_info=True)
        raise


def run_pipeline_train_only(**context):
    """Run training only on historical data.

    Useful for model retraining without immediate scoring.
    """
    execution_date = context["execution_date"].date()
    logger.info("Running train-only")

    ml_predict_path = Path(__file__).parent
    if str(ml_predict_path) not in sys.path:
        sys.path.insert(0, str(ml_predict_path))

    from config import load_config
    from train import EarlyExitAfterDiagnosticsLogs, train_pipeline
    from datetime import timedelta

    try:
        config = load_config()
        logger.info("Configuration loaded successfully")

        history_days = config.model.history_days
        history_start = execution_date - timedelta(days=history_days)

        logger.info(f"Training on data from {history_start} to {execution_date}")
        try:
            train_outputs = train_pipeline(config, start_date=history_start, end_date=execution_date)
        except EarlyExitAfterDiagnosticsLogs as exc:
            logger.info("Train task intentionally stopped after diagnostics logs: %s", exc)
            return {
                "status": "stopped_after_diagnostics_logs",
                "execution_date": execution_date.isoformat(),
            }

        logger.info(
            f"Training complete. Model: {train_outputs.model_path}. "
            f"Metrics: {train_outputs.metrics_path}"
        )

        return {
            "model_path": str(train_outputs.model_path),
            "metrics_path": str(train_outputs.metrics_path),
            "feature_frame_path": str(train_outputs.feature_frame_path),
        }

    except Exception as exc:
        logger.error(f"Train task failed: {exc}", exc_info=True)
        raise


def run_failed_dags_rca_report(**context):
    """Generate RCA report for previously failed DAG runs from error-log table."""
    execution_date = context["execution_date"].date()
    logger.info("Running failed-runs RCA report for execution date: %s", execution_date)

    ml_predict_path = Path(__file__).parent
    if str(ml_predict_path) not in sys.path:
        sys.path.insert(0, str(ml_predict_path))

    from config import load_config, load_dev_config
    from rca_failed_dags_report import generate_failed_dags_rca_report

    try:
        config = load_config()
        dev_cfg = load_dev_config()
        lookback_days = int(dev_cfg.get("rca_failed_runs_lookback_days", 7))

        outputs = generate_failed_dags_rca_report(
            config=config,
            target_date=execution_date,
            lookback_days=lookback_days,
        )

        logger.info(
            "Failed-runs RCA outputs: csv=%s md=%s html=%s rows=%s",
            outputs.csv_path,
            outputs.markdown_path,
            outputs.html_path,
            outputs.rows,
        )

        return {
            "failed_runs_rca_csv": str(outputs.csv_path),
            "failed_runs_rca_markdown": str(outputs.markdown_path),
            "failed_runs_rca_html": str(outputs.html_path) if outputs.html_path else None,
            "rows": outputs.rows,
            "lookback_days": lookback_days,
        }

    except Exception as exc:
        logger.error("Failed-runs RCA task failed: %s", exc, exc_info=True)
        raise


# --- Task Definitions ---

task_check_dependencies_train = PythonOperator(
    task_id="check_dependencies",
    python_callable=check_dependencies,
    provide_context=True,
    doc_md="Validate required Python dependencies before running train-only task.",
    dag=dag_train_only,
)

task_train_only = PythonOperator(
    task_id="run_ml_pipeline_train_only",
    python_callable=run_pipeline_train_only,
    provide_context=True,
    doc_md="Train model on historical data only.",
    dag=dag_train_only,
)

task_check_dependencies_predict = PythonOperator(
    task_id="check_dependencies",
    python_callable=check_dependencies,
    provide_context=True,
    doc_md="Validate required Python dependencies before running predict-only task.",
    dag=dag_predict_only,
)

task_predict_only = PythonOperator(
    task_id="run_ml_pipeline_predict_only",
    python_callable=run_pipeline_predict_only,
    provide_context=True,
    doc_md="Score today's runs (assumes model is already trained) and generate report.",
    dag=dag_predict_only,
)

task_check_dependencies_failed_rca = PythonOperator(
    task_id="check_dependencies",
    python_callable=check_dependencies,
    provide_context=True,
    doc_md="Validate required Python dependencies before running failed-runs RCA report task.",
    dag=dag_failed_rca,
)

task_failed_dags_rca_report = PythonOperator(
    task_id="run_failed_dags_rca_report",
    python_callable=run_failed_dags_rca_report,
    provide_context=True,
    doc_md="Generate RCA report for previously failed DAG runs from JONATHON_DAG_ERROR_LOG_RECORDS.",
    dag=dag_failed_rca,
)

# DAG flow: train-only
task_train_only.set_upstream(task_check_dependencies_train)

# DAG flow: predict-only
task_predict_only.set_upstream(task_check_dependencies_predict)

# DAG flow: failed-runs RCA report
task_failed_dags_rca_report.set_upstream(task_check_dependencies_failed_rca)
