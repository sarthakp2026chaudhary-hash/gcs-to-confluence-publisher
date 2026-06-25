# confluence_sources.py

"""Source adapters for the Confluence publisher.

The publisher's only contract with its data backend is :class:`Source`.

Phase A scaffolded :class:`GCSSource`.
Phase B made GCS auth-aware (SA impersonation + quota_project via
:class:`BigQueryConfig`).
Phase D added :meth:`Source.read_text` for the MD / JSON renderers.
Phase E makes :class:`BQSource` real — it uses ``google.cloud.bigquery`` with
the same impersonation + quota_project model as GCS.

The credentials helper :func:`_resolve_credentials` is shared between the GCS
and BQ client builders so both honor the same auth model.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Protocol
from urllib.parse import urlparse

import pandas as pd
from google.cloud import storage

from config import BigQueryConfig

logger = logging.getLogger(__name__)


class Source(Protocol):
    """Reads data from a backing store.

    Implementations:
        - :class:`GCSSource` — CSV / text objects under a configured GCS base URI.
        - :class:`BQSource`  — BigQuery tables.

    ``read_csv`` returns a DataFrame; ``read_text`` returns raw UTF-8 text. The
    runtime picks the right method based on the artefact's ``render_kind``:
    CSV uses ``read_csv``; MD and JSON use ``read_text``.
    """

    def read_csv(self, target: str) -> pd.DataFrame: ...

    def read_text(self, target: str) -> str: ...


class GCSSource:
    """Reads CSV / text artefacts from GCS, honoring SA impersonation + quota project.

    ``target`` is the object name relative to the configured base URI. For a
    base URI of ``gs://bucket/grid_search_ml/`` and a target of
    ``outputs/failed_runs_rca_20260612.csv`` the fetched object is
    ``gs://bucket/grid_search_ml/outputs/failed_runs_rca_20260612.csv``.
    """

    def __init__(
        self,
        base_uri: str,
        *,
        bq_config: BigQueryConfig | None = None,
    ) -> None:
        parsed = urlparse(base_uri)
        if parsed.scheme != "gs":
            raise ValueError(f"Invalid GCS URI: {base_uri}")
        self.base_uri = base_uri.rstrip("/")
        self._bucket_name = parsed.netloc
        self._base_prefix = parsed.path.lstrip("/").rstrip("/")
        self._client = _build_storage_client(bq_config)
        self._bucket = self._client.bucket(self._bucket_name)

    def read_csv(self, target: str) -> pd.DataFrame:
        text = self.read_text(target)
        return pd.read_csv(io.StringIO(text))

    def read_text(self, target: str) -> str:
        object_name = (
            f"{self._base_prefix}/{target}" if self._base_prefix else target
        )
        logger.info("GCS source read: gs://%s/%s", self._bucket_name, object_name)
        blob = self._bucket.blob(object_name)
        if not blob.exists():
            raise FileNotFoundError(
                f"GCS object not found: gs://{self._bucket_name}/{object_name}"
            )
        return blob.download_as_text(encoding="utf-8")

    def list_under(self, prefix: str) -> list[str]:
        """Return object names under ``{base_prefix}/{prefix}``.

        Used by ``check_dependencies`` to probe GCS reachability — at least one
        object found under the expected prefix means auth + path are correct.
        """
        full_prefix = (
            f"{self._base_prefix}/{prefix}".strip("/")
            if self._base_prefix
            else prefix.lstrip("/")
        )
        return [
            blob.name
            for blob in self._client.list_blobs(
                self._bucket_name, prefix=full_prefix, max_results=5
            )
        ]

    def try_read_text(self, target: str) -> str | None:
        """Read ``target`` if it exists, otherwise return ``None``.

        Used for Phase F state markers — a missing marker is a normal "first
        publish" condition, not an error.
        """
        object_name = (
            f"{self._base_prefix}/{target}" if self._base_prefix else target
        )
        blob = self._bucket.blob(object_name)
        if not blob.exists():
            return None
        return blob.download_as_text(encoding="utf-8")

    def write_text(self, target: str, content: str) -> None:
        """Upload ``content`` as a UTF-8 text blob at ``target``.

        Used for Phase F state markers — JSON sidecars under
        ``confluence_state/``. Overwrites any existing object.
        """
        object_name = (
            f"{self._base_prefix}/{target}" if self._base_prefix else target
        )
        blob = self._bucket.blob(object_name)
        blob.upload_from_string(
            content, content_type="application/json; charset=utf-8"
        )
        logger.info("GCS source write: gs://%s/%s", self._bucket_name, object_name)


class BQSource:
    """Reads CSV-equivalent tabular data from BigQuery.

    ``target`` is a fully-qualified table identifier (``project.dataset.table``).
    ``read_csv`` runs ``SELECT *`` against the target table and returns a
    ``pandas.DataFrame`` with the same column structure as the CSV would have.
    The renderer / publisher / page hierarchy downstream are unchanged — that
    invariance is what Phase E proves.

    ``read_text`` is not implemented for BQ: markdown and JSON artefacts in the
    current registry live in GCS, not BQ. If a future artefact needs JSON from
    a BQ JSON column, override here.
    """

    def __init__(self, *, bq_config: BigQueryConfig | None = None) -> None:
        self._bq_config = bq_config
        self._client: Any | None = None  # lazy: avoid BQ client cost when no BQ specs

    def read_csv(self, target: str) -> pd.DataFrame:
        client = self._get_client()
        query = f"SELECT * FROM `{target}`"
        logger.info("BQ source read: %s", target)
        return client.query(query).result().to_dataframe(create_bqstorage_client=False)

    def read_text(self, target: str) -> str:
        raise NotImplementedError(
            "BQSource.read_text is not implemented — MD / JSON artefacts in the "
            f"current registry live in GCS, not BQ. Target was {target!r}."
        )

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = _build_bigquery_client(self._bq_config)
        return self._client


def _build_storage_client(bq_config: BigQueryConfig | None) -> storage.Client:
    """Build a ``storage.Client`` honoring impersonation + quota project."""
    credentials, quota_project = _resolve_credentials(bq_config)
    kwargs: dict[str, Any] = {}
    if quota_project:
        kwargs["project"] = quota_project
    if credentials is not None:
        kwargs["credentials"] = credentials
    return storage.Client(**kwargs)


def _build_bigquery_client(bq_config: BigQueryConfig | None) -> Any:
    """Build a ``bigquery.Client`` honoring impersonation + quota project."""
    from google.cloud import bigquery  # deferred: only paid for if BQSource is used

    credentials, quota_project = _resolve_credentials(bq_config)
    kwargs: dict[str, Any] = {}
    if quota_project:
        kwargs["project"] = quota_project
    if credentials is not None:
        kwargs["credentials"] = credentials
    return bigquery.Client(**kwargs)


def _resolve_credentials(
    bq_config: BigQueryConfig | None,
) -> tuple[Any | None, str | None]:
    """Return ``(credentials, quota_project_id)``.

    Three modes:
        - No ``bq_config``                              → ``(None, None)`` — ADC, no quota_project.
        - ``bq_config`` without ``target_service_account`` → ``(None, quota_project)`` — ADC + quota_project.
        - ``bq_config`` with ``target_service_account``    → ``(impersonated_credentials, quota_project)``.

    The same model is reused by both GCS and BQ client builders.
    """
    if bq_config is None:
        return None, None

    quota_project = bq_config.quota_project_id

    if not bq_config.target_service_account:
        return None, quota_project

    from google.auth import default as _adc_default
    from google.auth import impersonated_credentials

    source_credentials, _ = _adc_default()
    target_credentials = impersonated_credentials.Credentials(
        source_credentials=source_credentials,
        target_principal=bq_config.target_service_account,
        target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
        lifetime=bq_config.token_lifetime,
    )
    logger.info(
        "Cloud client impersonating %s (quota_project=%s)",
        bq_config.target_service_account,
        bq_config.quota_project_id,
    )
    return target_credentials, quota_project
