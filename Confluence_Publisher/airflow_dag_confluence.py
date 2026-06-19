# airflow_dag_confluence.py

"""DAG definition for publishing GCS / BQ artefacts to Confluence.

One DAG, two tasks (dependency check + publish), manual trigger by default.

``check_dependencies`` validates required Python packages, probes GCS
reachability under the configured base URI, and probes Confluence reachability
via ``get_space``. Optional packages (``markdown``) are logged but not fatal.

``run_publish`` loads config, retrieves the Confluence token via the Airflow
Connection referenced by ``confluence_auth_connection_id``, shifts
``target_date`` back by ``confluence_config.lookback_days``, and delegates to
``confluence_runtime.publish_all``. The result dict (published /
skipped_unchanged / errors / target_date) is returned via XCom for downstream
DAGs.

The GCS client honors :class:`BigQueryConfig` impersonation + quota_project.
Confluence errors are sanitized at the publisher layer (no token in any log
line). On task failure :func:`_on_failure_callback` logs a structured one-liner
— wrap it for real alerting (Slack / PagerDuty) when the project adopts a
convention.

Flip ``schedule_interval`` to :data:`_PROPOSED_SCHEDULE` once the daily ETL
window is confirmed with data-eng.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from airflow.models import DAG
from airflow.operators.python import PythonOperator

# Add the DAG folder to sys.path so the module-level ``naming`` import resolves
# at DAG-parse time (Composer's scheduler imports DAG files standalone).
sys.path.insert(0, str(Path(__file__).parent))

from naming import PUBLISH_DAG_ID  # noqa: E402

logger = logging.getLogger(__name__)


# Phase F: a daily 06:00 UTC schedule is proposed but not active. Flipping
# ``schedule_interval`` from ``None`` to ``_PROPOSED_SCHEDULE`` would make
# this the FIRST scheduled DAG in this folder — confirm the ETL window with
# the data-eng team before promoting.
_PROPOSED_SCHEDULE = "0 6 * * *"


def _on_failure_callback(context: dict[str, Any]) -> None:
    """Log a structured one-liner on task failure (Phase F).

    Real alerting (email, Slack, PagerDuty) is intentionally out of scope —
    add an on-failure handler that wraps this one when the project adopts a
    convention.
    """
    dag_run = context.get("dag_run")
    task_instance = context.get("task_instance")
    logger.error(
        "Confluence DAG task failed: dag_run_id=%s task_id=%s exception=%s",
        getattr(dag_run, "run_id", "?"),
        getattr(task_instance, "task_id", "?"),
        context.get("exception"),
    )


default_args = {
    "owner": "data-eng",
    "depends_on_past": False,
    "start_date": datetime(2026, 5, 6),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": _on_failure_callback,
}

dag_confluence_publish = DAG(
    dag_id=PUBLISH_DAG_ID,
    default_args=default_args,
    schedule_interval=None,  # Manual trigger; flip to _PROPOSED_SCHEDULE for daily.
    catchup=False,
    doc_md="""
# GCS → Confluence Publisher

Reads ML pipeline artefacts from GCS (or BigQuery, per registry entry) and
posts them as Confluence pages so downstream consumers (Copilot agent / ML
retraining loop) can read them without GCS or BigQuery access.

**Registry:** `confluence_runtime.py:REGISTRY` — add a row to publish a new
artefact. Supports CSV, Markdown, and JSON render kinds.

**Configuration:** `dev_config.json` `confluence_*` keys hold identifiers
only. The API token / PAT lives in an Airflow Connection referenced by
`confluence_auth_connection_id`.

**Hierarchy:** each artefact type gets a sub-parent page (auto-created under
the configured root `parent_page_id`); each run posts a dated child below it.

**Idempotency:**
- Title-based update: re-running updates the existing page (Confluence bumps
  the version).
- Content-hash skip: if the source bytes hash to the same value as the last
  successful publish (state markers in `confluence_state/`), the artefact is
  skipped entirely — no Confluence call is made.

**Lookback:** `target_date = execution_date - lookback_days` so artefacts
dated within the last N days are skipped (they may still be churning in the
upstream ML pipeline).
""",
)


def check_dependencies_confluence(**context: Any) -> dict[str, Any]:
    """Fail fast on missing deps, unreachable GCS, or unreachable Confluence.

    Three layers of check, in order:
        1. Required Python modules import successfully (catches missing
           ``atlassian-python-api`` before any real work).
        2. GCS bucket is reachable under the configured base URI — at least
           the bucket exists and the worker SA (or impersonated SA) can list
           objects under ``outputs/``.
        3. Confluence space is reachable — ``get_space`` returns without error.

    Any failure raises ``RuntimeError`` with a message that names the failing
    surface (connection id / GCS URI), never the credential value.
    """
    required_modules = {
        "pandas": "pandas",
        "google.cloud.storage": "google-cloud-storage",
        "atlassian": "atlassian-python-api",
    }
    optional_modules = {
        "markdown": "markdown",  # used by render_markdown — falls back to <pre> if absent
    }

    missing_required: list[str] = []
    for module_name, package_name in required_modules.items():
        try:
            __import__(module_name)
        except Exception:
            missing_required.append(package_name)

    missing_optional: list[str] = []
    for module_name, package_name in optional_modules.items():
        try:
            __import__(module_name)
        except Exception:
            missing_optional.append(package_name)

    if missing_optional:
        logger.warning(
            "Optional packages missing (renderer falls back gracefully): %s",
            ", ".join(missing_optional),
        )

    if missing_required:
        raise RuntimeError(
            "Missing required Python packages in runtime: "
            f"{', '.join(missing_required)}. "
            "Install via Composer → Environment → PyPI packages."
        )

    ml_predict_path = Path(__file__).parent
    if str(ml_predict_path) not in sys.path:
        sys.path.insert(0, str(ml_predict_path))

    from airflow.hooks.base import BaseHook

    from config import load_bq_config, load_config, load_confluence_config
    from confluence_publisher import ConfluencePublisher
    from confluence_sources import GCSSource

    config = load_config()
    bq_config = load_bq_config()
    confluence_config = load_confluence_config()

    if not config.paths.output_gcs_uri:
        raise RuntimeError(
            "paths.output_gcs_uri is not set in dev_config.json — "
            "cannot probe GCS."
        )

    try:
        gcs_source = GCSSource(
            base_uri=config.paths.output_gcs_uri,
            bq_config=bq_config,
        )
        objects = gcs_source.list_under("outputs/")
    except Exception as exc:
        raise RuntimeError(
            f"GCS probe failed for base_uri={config.paths.output_gcs_uri!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if not objects:
        logger.warning(
            "GCS probe: 0 objects under %s/outputs/ — no artefacts to publish yet. "
            "Verify the bucket path is correct.",
            config.paths.output_gcs_uri,
        )

    conn = BaseHook.get_connection(confluence_config.auth_connection_id)
    publisher = ConfluencePublisher(
        config=confluence_config,
        username=conn.login or "",
        password=conn.password or "",
    )
    publisher.probe()

    logger.info(
        "Dependency + reachability probes passed: gcs_objects_found=%s "
        "confluence_space=%s connection_id=%s",
        len(objects),
        confluence_config.space_key,
        confluence_config.auth_connection_id,
    )
    return {
        "dependency_check": "passed",
        "gcs_probe_objects_found": len(objects),
    }


def run_publish(**context: Any) -> dict[str, Any]:
    """Publish registered GCS artefacts to Confluence.

    Phase F: ``target_date`` is shifted back by ``confluence_config.lookback_days``
    (default 2) so the publisher skips artefacts dated "too recent" — these may
    still be churning in the upstream pipeline. The DAG's logical_date (the run
    date) is kept separately in ``run_context`` for footer attribution.
    """
    execution_date = context["execution_date"].date()
    logger.info("Running Confluence publish: execution_date=%s", execution_date)

    ml_predict_path = Path(__file__).parent
    if str(ml_predict_path) not in sys.path:
        sys.path.insert(0, str(ml_predict_path))

    from airflow.hooks.base import BaseHook

    from config import load_bq_config, load_config, load_confluence_config
    from confluence_runtime import publish_all

    try:
        config = load_config()
        bq_config = load_bq_config()
        confluence_config = load_confluence_config()
        logger.info(
            "Configuration loaded: connection_id=%s flavor=%s lookback_days=%s",
            confluence_config.auth_connection_id,
            confluence_config.flavor,
            confluence_config.lookback_days,
        )

        target_date = execution_date - timedelta(
            days=confluence_config.lookback_days
        )
        logger.info(
            "Shifted target_date for publish: execution_date=%s lookback_days=%s -> target_date=%s",
            execution_date,
            confluence_config.lookback_days,
            target_date,
        )

        conn = BaseHook.get_connection(confluence_config.auth_connection_id)
        username = conn.login or ""
        password = conn.password or ""

        dag_run = context.get("dag_run")
        run_context = {
            "dag_run_id": str(dag_run.run_id) if dag_run is not None else "",
            "logical_date": execution_date.isoformat(),
        }

        result = publish_all(
            target_date=target_date,
            config=config,
            confluence_config=confluence_config,
            confluence_username=username,
            confluence_password=password,
            bq_config=bq_config,
            run_context=run_context,
        )

        logger.info(
            "Confluence publish complete: published=%s skipped_unchanged=%s errors=%s target_date=%s",
            len(result["published"]),
            len(result["skipped_unchanged"]),
            len(result["errors"]),
            result["target_date"],
        )

        for err in result["errors"]:
            logger.error("Publish error: %s", err)

        return result

    except Exception as exc:
        logger.error("Confluence publish task failed: %s", exc, exc_info=True)
        raise


# --- Task Definitions ---

task_check_dependencies_confluence = PythonOperator(
    task_id="check_dependencies",
    python_callable=check_dependencies_confluence,
    provide_context=True,
    doc_md="Validate deps + probe GCS reachability + probe Confluence reachability.",
    dag=dag_confluence_publish,
)

task_run_publish = PythonOperator(
    task_id="run_publish",
    python_callable=run_publish,
    provide_context=True,
    doc_md="Read registered GCS artefacts and publish each as a Confluence page.",
    dag=dag_confluence_publish,
)

# DAG flow: GCS → Confluence publish
task_run_publish.set_upstream(task_check_dependencies_confluence)
