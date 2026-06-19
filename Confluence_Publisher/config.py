# config.py — minimal config for the Confluence publisher DAG.
#
# This folder deploys to Composer independently of GridSearch_ML/. It does NOT
# import anything from that folder. The publisher_config.json next to this
# file must carry:
#   - output_gcs_uri (must match what GridSearch_ML writes to)
#   - target_service_account / quota_project_id / token_lifetime (auth)
#   - confluence_* keys (identifiers only; token via Airflow Connection)
#
# Drift warning: output_gcs_uri here MUST stay in sync with the value in
# GridSearch_ML/dev_config.json. If GridSearch_ML's bucket prefix changes,
# this folder's config must be updated to match — otherwise the publisher
# reads from an empty / wrong path.

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


_PUBLISHER_CONFIG_PATH = Path(__file__).parent / "publisher_config.json"


def load_publisher_config() -> dict[str, Any]:
    """Load configuration from publisher_config.json located next to this file."""
    with _PUBLISHER_CONFIG_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


_publisher_cfg_cache: dict[str, Any] | None = None


def _publisher_cfg() -> dict[str, Any]:
    """Return a cached copy of publisher_config.json."""
    global _publisher_cfg_cache
    if _publisher_cfg_cache is None:
        _publisher_cfg_cache = load_publisher_config()
    return _publisher_cfg_cache


@dataclass(frozen=True)
class PathsConfig:
    output_gcs_uri: str | None = None


@dataclass(frozen=True)
class BigQueryConfig:
    """Auth configuration for GCS + (optional) BigQuery clients.

    The publisher uses SA impersonation when ``target_service_account`` is set,
    and routes quota / billing to ``quota_project_id`` when set. ``project_id``
    and ``dataset_id`` are accepted for symmetry with GridSearch_ML but the
    publisher does not require them — the BQSource resolves the BQ table from
    the registry's ``path_template`` directly.
    """

    project_id: str = ""
    dataset_id: str = ""
    target_service_account: str | None = None
    quota_project_id: str | None = None
    token_lifetime: int = 3600


@dataclass(frozen=True)
class ConfluenceConfig:
    """Confluence publishing configuration.

    Identifiers only. The API token / PAT lives in an Airflow Connection
    referenced by ``auth_connection_id`` and is fetched at task runtime via
    ``BaseHook.get_connection``. Nothing in this dataclass is a secret —
    matching the pattern used by GridSearch_ML/dev_config.json (no password
    in the JSON; credentials come via Airflow Connection or ADC).
    """

    base_url: str
    space_key: str
    parent_page_id: str
    flavor: str  # "cloud" | "server"
    auth_connection_id: str
    row_cap: int = 5000
    lookback_days: int = 2
    wide_cell_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class AppConfig:
    """Slim AppConfig — only the paths block is needed by the publisher."""

    paths: PathsConfig


def load_config() -> AppConfig:
    """Load minimal AppConfig (paths only) from publisher_config.json."""
    dcfg = _publisher_cfg()
    return AppConfig(
        paths=PathsConfig(
            output_gcs_uri=str(dcfg.get("output_gcs_uri", "")).strip() or None,
        ),
    )


def load_bq_config() -> BigQueryConfig:
    """Load BigQueryConfig from publisher_config.json.

    All fields are optional. With ``target_service_account`` unset (default),
    the publisher uses ADC. With it set, the GCS / BQ clients honor
    impersonation. ``quota_project_id`` routes billing to a separate project
    when set.
    """
    dcfg = _publisher_cfg()

    target_sa = str(dcfg.get("target_service_account", "")).strip() or None
    quota_project = str(dcfg.get("quota_project_id", "")).strip() or None
    token_lifetime = int(dcfg.get("token_lifetime", 3600))

    return BigQueryConfig(
        project_id=str(dcfg.get("project_id", "")).strip(),
        dataset_id=str(dcfg.get("dataset_id", "")).strip(),
        target_service_account=target_sa,
        quota_project_id=quota_project,
        token_lifetime=token_lifetime,
    )


def load_confluence_config() -> ConfluenceConfig:
    """Load Confluence publishing configuration from publisher_config.json.

    Required keys:
        - confluence_base_url
        - confluence_space_key
        - confluence_parent_page_id
        - confluence_auth_connection_id

    Optional keys (with defaults):
        - confluence_flavor             (default "cloud")
        - confluence_row_cap            (default 5000)
        - confluence_lookback_days      (default 2)
        - confluence_wide_cell_columns  (default ())
    """
    dcfg = _publisher_cfg()

    base_url = str(dcfg.get("confluence_base_url", "")).strip()
    if not base_url:
        raise ValueError("confluence_base_url is not set in publisher_config.json.")

    space_key = str(dcfg.get("confluence_space_key", "")).strip()
    if not space_key:
        raise ValueError("confluence_space_key is not set in publisher_config.json.")

    parent_page_id = str(dcfg.get("confluence_parent_page_id", "")).strip()
    if not parent_page_id:
        raise ValueError("confluence_parent_page_id is not set in publisher_config.json.")

    auth_connection_id = str(dcfg.get("confluence_auth_connection_id", "")).strip()
    if not auth_connection_id:
        raise ValueError("confluence_auth_connection_id is not set in publisher_config.json.")

    flavor = str(dcfg.get("confluence_flavor", "cloud")).strip().lower()
    if flavor not in {"cloud", "server"}:
        raise ValueError(
            f"confluence_flavor must be 'cloud' or 'server', got {flavor!r}."
        )

    row_cap = int(dcfg.get("confluence_row_cap", 5000))
    lookback_days = int(dcfg.get("confluence_lookback_days", 2))
    wide_cell_columns = tuple(dcfg.get("confluence_wide_cell_columns", []) or [])

    return ConfluenceConfig(
        base_url=base_url.rstrip("/"),
        space_key=space_key,
        parent_page_id=parent_page_id,
        flavor=flavor,
        auth_connection_id=auth_connection_id,
        row_cap=row_cap,
        lookback_days=lookback_days,
        wide_cell_columns=wide_cell_columns,
    )


def date_to_str(dt: date) -> str:
    return dt.strftime("%Y%m%d")
