# `confluence_runtime.py`

Source path: `Confluence_Publisher/confluence_runtime.py`

```python
# confluence_runtime.py

"""Artefact registry + orchestration glue for the Confluence publisher.

The registry declares one row per artefact to publish. The orchestration loop
reads each artefact once, hashes its content, skips unchanged artefacts via a
sidecar state marker in GCS, then renders + publishes the rest.

Phase A scaffolded the loop with one entry.
Phase C added the parent + dated-children hierarchy.
Phase D expanded the registry to 6 entries spanning CSV / MD / JSON.
Phase E added BQSource as a real source-kind option (flip via the comment
block at the top of REGISTRY).
Phase F added:
    - Content-hash idempotency: artefacts whose source bytes are unchanged
      since the last successful publish skip rendering AND the Confluence
      round-trip entirely. State markers live as JSON sidecars under
      ``{output_gcs_uri}/confluence_state/``.
    - XCom dict gains a ``skipped_unchanged`` list so downstream tasks /
      humans can tell "skipped because nothing to do" from "skipped because
      it failed."
    - Per-artefact failures still don't abort the run — each error is caught
      and collected with a sanitized message.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from config import AppConfig, BigQueryConfig, ConfluenceConfig, date_to_str
from confluence_publisher import ConfluencePublisher, _sanitize
from confluence_renderer import render_csv_table, render_json_table, render_markdown
from confluence_sources import BQSource, GCSSource, Source

logger = logging.getLogger(__name__)


_STATE_PREFIX = "confluence_state"


@dataclass(frozen=True)
class ArtefactSpec:
    """One row of the publish registry.

    ``path_template`` placeholders:
        ``{date}``     → compact date, e.g. ``20260612``
        ``{iso_date}`` → ISO date,    e.g. ``2026-06-12``

    ``title_template`` accepts the same placeholders.

    ``parent_title`` is the artefact-type sub-parent page. Phase C's hierarchy
    creates it on demand under the config's ``parent_page_id`` and the run's
    dated child sits below it.
    """

    path_template: str
    title_template: str
    parent_title: str
    render_kind: str  # "csv" | "md" | "json"
    source_kind: str  # "gcs" | "bq"


# ----------------------------------------------------------------------------
# SMOKE-TEST REGISTRY (current state)
# ----------------------------------------------------------------------------
# Pinned to ONE specific file so we can verify the full pipe end-to-end
# against a single ~900 KB CSV before scaling up to the multi-artefact loop.
#
# The publisher will:
#   1. Read gs://<bucket>/outputs/grid_cv_predictions_20260612.csv
#   2. Render its rows as an XHTML table on a Confluence page titled
#      "Daily Predictions — 2026-06-12"
#   3. Create a "Daily Predictions" sub-parent page under the configured
#      confluence_parent_page_id, then post the dated child below it
#   4. Attach the raw CSV bytes to the dated child page (text/csv) for the
#      agent to download via the REST API
#
# To restore the full 6-artefact registry once the smoke succeeds, see git
# log for "feat: GCS to Confluence publisher Airflow DAG" — the original
# REGISTRY tuple is in that commit.
#
# Eventual goals (NOT implemented yet):
#   - Sweep + delete Confluence pages older than 7 days
#   - Auto-discover files for the last 2 days and skip ones already published
#   - These are tracked separately; the smoke test only proves the pipe works.
#
# Notes on the pinned path:
#   - path_template has NO {date}/{iso_date} placeholders, so the DAG's
#     target_date computation (execution_date - lookback_days) is a no-op
#     for this entry. You can trigger the DAG on any day and it will always
#     try this one file.
#   - title_template is also literal — no template substitution required.
# ----------------------------------------------------------------------------
REGISTRY: tuple[ArtefactSpec, ...] = (
    ArtefactSpec(
        path_template="outputs/grid_cv_predictions_20260612.csv",
        title_template="Daily Predictions — 2026-06-12",
        parent_title="Daily Predictions",
        render_kind="csv",
        source_kind="gcs",
    ),
)


@dataclass(frozen=True)
class _PublishedMarker:
    """Sidecar state for a previously-published artefact."""

    page_id: str
    content_hash: str
    target: str
    published_at: str


def publish_all(
    *,
    target_date: date,
    config: AppConfig,
    confluence_config: ConfluenceConfig,
    confluence_username: str,
    confluence_password: str,
    bq_config: BigQueryConfig | None = None,
    run_context: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Publish every registered artefact for ``target_date`` to Confluence.

    Returns:
        ``{"published": [...], "skipped_unchanged": [...], "errors": [...],
        "target_date": "2026-06-12"}``

    Per-artefact failures are caught and collected in ``errors`` (with messages
    passed through :func:`_sanitize`) so a single missing file or bad render
    does not abort the whole DAG run.

    Phase F: artefacts whose source bytes hash to the same value as the last
    successful publish are skipped entirely — no Confluence call is made.
    """
    if not config.paths.output_gcs_uri:
        raise ValueError(
            "paths.output_gcs_uri is not set in dev_config.json — "
            "cannot read artefacts from GCS."
        )

    gcs_source = GCSSource(
        base_uri=config.paths.output_gcs_uri,
        bq_config=bq_config,
    )
    bq_source = BQSource(bq_config=bq_config)  # lazy — free if no BQ specs
    publisher = ConfluencePublisher(
        config=confluence_config,
        username=confluence_username,
        password=confluence_password,
    )

    iso_date = target_date.isoformat()
    compact_date = date_to_str(target_date)
    generated_at = datetime.now(timezone.utc)

    published: list[dict[str, Any]] = []
    skipped_unchanged: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    subparent_cache: dict[str, str] = {}

    for spec in REGISTRY:
        target = spec.path_template.format(date=compact_date, iso_date=iso_date)
        title = spec.title_template.format(date=compact_date, iso_date=iso_date)

        try:
            # Rate-limit defense: 15-20s randomized pause between artefacts.
            # Mirrors the ``time.sleep(DELAY)`` pattern in the
            # JONATHON_HUB_CICD_TOOL_RESTFUL reference (confluence.py +
            # confl_page_create.py both gate each major Confluence operation
            # behind a 10-20s sleep). At ~6 artefacts per run, total added
            # latency is ~90-120 seconds — negligible vs. an unmetered DAG.
            time.sleep(random.randint(15, 20))

            source = _pick_source(
                spec, gcs_source=gcs_source, bq_source=bq_source
            )
            logger.info("Reading artefact: target=%s title=%r", target, title)

            data, content_hash = _read_and_hash(spec, source, target)
            state_key = _state_key_from_target(target)
            existing_marker = _load_state_marker(gcs_source, state_key)

            if (
                existing_marker is not None
                and existing_marker.content_hash == content_hash
            ):
                logger.info(
                    "Skipping unchanged artefact: %s (hash=%s page_id=%s)",
                    target,
                    content_hash[:12],
                    existing_marker.page_id,
                )
                skipped_unchanged.append(
                    {
                        "artefact": spec.path_template,
                        "target": target,
                        "page_id": existing_marker.page_id,
                        "content_hash": content_hash,
                    }
                )
                continue

            source_uri = f"{config.paths.output_gcs_uri.rstrip('/')}/{target}"
            body, size = _render_from_data(
                spec,
                data,
                title=title,
                source_uri=source_uri,
                generated_at=generated_at,
                row_cap=confluence_config.row_cap,
                wide_cell_columns=confluence_config.wide_cell_columns,
                run_context=run_context,
            )

            if spec.parent_title not in subparent_cache:
                subparent_cache[spec.parent_title] = publisher.ensure_subparent(
                    title=spec.parent_title
                )
            sub_parent_id = subparent_cache[spec.parent_title]

            page_id = publisher.create_or_update(
                title=title,
                body_xhtml=body,
                parent_page_id=sub_parent_id,
            )

            attachment_content, attachment_content_type, attachment_filename = (
                _attachment_for(spec, data, target)
            )
            publisher.attach_source(
                page_id=page_id,
                filename=attachment_filename,
                content=attachment_content,
                content_type=attachment_content_type,
            )

            _write_state_marker(
                gcs_source,
                state_key,
                _PublishedMarker(
                    page_id=page_id,
                    content_hash=content_hash,
                    target=target,
                    published_at=generated_at.isoformat(),
                ),
            )

            published.append(
                {
                    "artefact": spec.path_template,
                    "target": target,
                    "title": title,
                    "page_id": page_id,
                    "size": size,
                    "content_hash": content_hash,
                    "render_kind": spec.render_kind,
                    "parent_title": spec.parent_title,
                    "attachment_filename": attachment_filename,
                    "attachment_content_type": attachment_content_type,
                }
            )
        except Exception as exc:
            logger.exception("Failed to publish artefact: %s", target)
            errors.append(
                {
                    "artefact": spec.path_template,
                    "target": target,
                    "error": f"{type(exc).__name__}: {_sanitize(str(exc))}",
                }
            )

    return {
        "published": published,
        "skipped_unchanged": skipped_unchanged,
        "errors": errors,
        "target_date": iso_date,
    }


def _pick_source(
    spec: ArtefactSpec,
    *,
    gcs_source: GCSSource,
    bq_source: BQSource,
) -> Source:
    if spec.source_kind == "gcs":
        return gcs_source
    if spec.source_kind == "bq":
        return bq_source
    raise ValueError(f"Unknown source_kind: {spec.source_kind!r}")


def _read_and_hash(
    spec: ArtefactSpec,
    source: Source,
    target: str,
) -> tuple[Any, str]:
    """Read the artefact and compute a stable content hash.

    For CSV: read as DataFrame; hash the canonical CSV serialization
    (``to_csv(index=False)``) so the hash is stable across re-reads and works
    identically whether the source was GCS-text or a BQ result-set.

    For MD / JSON: read as text; hash the raw text.
    """
    if spec.render_kind == "csv":
        df = source.read_csv(target)
        canonical = df.to_csv(index=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return df, digest

    if spec.render_kind in ("md", "json"):
        text = source.read_text(target)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return text, digest

    raise ValueError(f"Unknown render_kind: {spec.render_kind!r}")


def _render_from_data(
    spec: ArtefactSpec,
    data: Any,
    *,
    title: str,
    source_uri: str,
    generated_at: datetime,
    row_cap: int,
    wide_cell_columns: tuple[str, ...],
    run_context: dict[str, str] | None,
) -> tuple[str, int]:
    """Render pre-read data. Returns ``(body_xhtml, size)``.

    ``size`` is rows for CSV and bytes for MD / JSON.
    """
    if spec.render_kind == "csv":
        df: pd.DataFrame = data
        body = render_csv_table(
            df,
            title=title,
            source_uri=source_uri,
            generated_at=generated_at,
            row_cap=row_cap,
            wide_cell_columns=wide_cell_columns,
            run_context=run_context,
        )
        return body, len(df)

    if spec.render_kind == "md":
        text: str = data
        body = render_markdown(
            text,
            title=title,
            source_uri=source_uri,
            generated_at=generated_at,
            run_context=run_context,
        )
        return body, len(text)

    if spec.render_kind == "json":
        text = data
        obj = json.loads(text)
        body = render_json_table(
            obj,
            title=title,
            source_uri=source_uri,
            generated_at=generated_at,
            run_context=run_context,
        )
        return body, len(text)

    raise ValueError(f"Unknown render_kind: {spec.render_kind!r}")


def _state_key_from_target(target: str) -> str:
    """Build a stable filename-safe key from an artefact target string."""
    return target.replace("/", "_").replace(".", "_").replace(" ", "_")


def _load_state_marker(
    gcs_source: GCSSource,
    state_key: str,
) -> _PublishedMarker | None:
    """Load the JSON sidecar for ``state_key`` if it exists. Otherwise None."""
    text = gcs_source.try_read_text(f"{_STATE_PREFIX}/{state_key}.json")
    if text is None:
        return None
    try:
        payload = json.loads(text)
        return _PublishedMarker(
            page_id=str(payload["page_id"]),
            content_hash=str(payload["content_hash"]),
            target=str(payload.get("target", "")),
            published_at=str(payload.get("published_at", "")),
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning(
            "Corrupted state marker %s — re-publishing. (%s)", state_key, exc
        )
        return None


def _write_state_marker(
    gcs_source: GCSSource,
    state_key: str,
    marker: _PublishedMarker,
) -> None:
    """Upload the JSON sidecar for ``state_key`` (overwrites any existing)."""
    payload = json.dumps(
        {
            "page_id": marker.page_id,
            "content_hash": marker.content_hash,
            "target": marker.target,
            "published_at": marker.published_at,
        },
        indent=2,
        sort_keys=True,
    )
    gcs_source.write_text(f"{_STATE_PREFIX}/{state_key}.json", payload)


def _attachment_for(
    spec: ArtefactSpec,
    data: Any,
    target: str,
) -> tuple[bytes, str, str]:
    """Build the attachment payload for an artefact.

    Returns ``(content_bytes, content_type, display_filename)``.

    The attachment is what the downstream agent downloads via the Confluence
    REST API and parses natively (instead of scraping the rendered XHTML
    table). For CSV-rendered artefacts the canonical serialization
    ``df.to_csv(index=False)`` is used — this is the same form that
    :func:`_read_and_hash` hashes, so the attached file matches the content
    hash regardless of whether the source was GCS text or a BQ result set.
    """
    filename = target.rsplit("/", 1)[-1]

    if spec.render_kind == "csv":
        df: pd.DataFrame = data
        return (
            df.to_csv(index=False).encode("utf-8"),
            "text/csv",
            filename,
        )

    if spec.render_kind == "md":
        text: str = data
        return text.encode("utf-8"), "text/markdown", filename

    if spec.render_kind == "json":
        text = data
        return text.encode("utf-8"), "application/json", filename

    raise ValueError(f"Unknown render_kind: {spec.render_kind!r}")
```
