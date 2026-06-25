# `confluence_runtime.py`

Source path: `confluence_publisher_2/confluence_runtime.py`

```python
# confluence_runtime.py

"""Artefact registry + orchestration glue for the Confluence publisher.

Phase G — production daily mode. Two entry points:

  :func:`publish_all` — discovers new (artefact, date) pairs from GCS for the
  last ``lookback_days``, skips any whose Confluence page already exists,
  optionally applies a per-artefact row filter, renders + publishes + attaches
  the rest. Per-train rolling artefacts (no date in path) use a content-hash
  skip via GCS sidecar markers (unchanged from Phase F).

  :func:`sweep_all` — deletes dated child pages under each artefact-type
  sub-parent whose title-date is older than ``retention_days``. Skips pages
  without a parseable date (rolling pages stay forever) and pages whose body
  lacks the ``auto_marker`` comment (humans have taken ownership).

Per-artefact failures are caught and collected so one bad file does not abort
the rest of the run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

import pandas as pd

from config import AppConfig, BigQueryConfig, ConfluenceConfig, date_to_str
from confluence_publisher import ConfluencePublisher, _sanitize
from confluence_renderer import (
    PAGE_BODY_HARD_LIMIT_BYTES,
    PAGE_BODY_WARN_BYTES,
    body_size_bytes,
    render_attachment_only_placeholder,
    render_csv_table,
    render_json_table,
    render_markdown,
)
from confluence_sources import BQSource, GCSSource, Source

logger = logging.getLogger(__name__)


_STATE_PREFIX = "confluence_state"

# Captures YYYYMMDD inside a filename like grid_cv_predictions_20260612.csv.
_FILENAME_DATE_PATTERN = re.compile(r"_(\d{8})\.")

# Captures the trailing ISO date in titles like "Daily Predictions — 2026-06-12".
# Used by the retention sweep to identify dated child pages.
_TITLE_DATE_PATTERN = re.compile(r"—\s*(\d{4}-\d{2}-\d{2})\s*$")


# ----------------------------------------------------------------------------
# Per-artefact row filters
# ----------------------------------------------------------------------------


def _predictions_risk_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only Med + High risk predictions; drop Low.

    The downstream Copilot agent doesn't need Low-risk rows — they're "this
    DAG is fine" noise. Med + High are the rows worth attention.

    Defensive: if the ``risk_bucket`` column is missing (schema drift),
    return the DataFrame untouched and log a warning rather than silently
    dropping every row.
    """
    if "risk_bucket" not in df.columns:
        logger.warning(
            "_predictions_risk_filter: 'risk_bucket' column missing — "
            "keeping all rows."
        )
        return df
    before = len(df)
    filtered = df[df["risk_bucket"].isin(["Med", "High"])]
    after = len(filtered)
    logger.info(
        "Risk filter: kept %s of %s rows (Med + High only).", after, before
    )
    return filtered


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtefactSpec:
    """One row of the publish registry.

    ``path_template`` placeholders:
        ``{date}``     → compact date, e.g. ``20260612``
        ``{iso_date}`` → ISO date,    e.g. ``2026-06-12``

    A path WITH a placeholder is a "dated" artefact — discovery globs GCS
    for matching files and produces one publish per date found in the
    lookback window. Each dated artefact gets its own Confluence page named
    ``"<title_template>"`` (with placeholders expanded).

    A path WITHOUT any placeholder is a "rolling" artefact — single page,
    overwritten each run if the content hash changes. Used for per-train
    artefacts that don't carry a date in their filename
    (``grid_cv_train_metrics.json``, ``grid_cv_feature_importance.csv``).

    ``row_filter`` is applied AFTER read, BEFORE hash + render. Lets a spec
    drop rows it considers noise (e.g. Low-risk predictions).
    """

    path_template: str
    title_template: str
    parent_title: str
    render_kind: str  # "csv" | "md" | "json"
    source_kind: str  # "gcs" | "bq"
    row_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = field(
        default=None, compare=False
    )


# To migrate an entry to BigQuery (Phase E):
#   1. Change ``source_kind`` to ``"bq"``.
#   2. Change ``path_template`` to a fully-qualified BQ table reference, e.g.
#      ``"<project>.<dataset>.<TABLE_PREFIX>_{date}"``.
#   3. Ensure the BQ table mirrors the original CSV's columns — renderer,
#      publisher, page hierarchy stay unchanged.
REGISTRY: tuple[ArtefactSpec, ...] = (
    # Daily, date-stamped (under outputs/) — discovery globs by prefix and
    # publishes one page per date found in the lookback window.
    ArtefactSpec(
        path_template="outputs/failed_runs_rca_{date}.csv",
        title_template="Failed Runs RCA — {iso_date}",
        parent_title="Failed Runs RCA",
        render_kind="csv",
        source_kind="gcs",
    ),
    ArtefactSpec(
        path_template="outputs/grid_cv_predictions_{date}.csv",
        title_template="Daily Predictions — {iso_date}",
        parent_title="Daily Predictions",
        render_kind="csv",
        source_kind="gcs",
        row_filter=_predictions_risk_filter,  # drop Low; keep Med + High
    ),
    ArtefactSpec(
        path_template="outputs/failed_runs_rca_report_{date}.md",
        title_template="Failed Runs RCA Report — {iso_date}",
        parent_title="Failed Runs RCA Reports",
        render_kind="md",
        source_kind="gcs",
    ),
    ArtefactSpec(
        path_template="outputs/report_{date}.md",
        title_template="Daily Report — {iso_date}",
        parent_title="Daily Reports",
        render_kind="md",
        source_kind="gcs",
    ),
    # Per-train, NOT date-stamped (under artefacts/) — overwritten on each
    # training run. ONE page per artefact, content-hash skip on re-runs.
    ArtefactSpec(
        path_template="artefacts/grid_cv_train_metrics.json",
        title_template="Latest Train Metrics",
        parent_title="Train Metrics",
        render_kind="json",
        source_kind="gcs",
    ),
    ArtefactSpec(
        path_template="artefacts/grid_cv_feature_importance.csv",
        title_template="Latest Feature Importance",
        parent_title="Feature Importance",
        render_kind="csv",
        source_kind="gcs",
    ),
)


@dataclass(frozen=True)
class _PublishedMarker:
    """Sidecar state for a previously-published rolling artefact."""

    page_id: str
    content_hash: str
    target: str
    published_at: str


# ----------------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------------


def _is_dated(spec: ArtefactSpec) -> bool:
    return "{date}" in spec.path_template or "{iso_date}" in spec.path_template


def _discover_pairs(
    *,
    registry: tuple[ArtefactSpec, ...],
    gcs_source: GCSSource,
    today: date,
    lookback_days: int,
) -> list[tuple[ArtefactSpec, date | None]]:
    """Return the list of (spec, date) pairs to consider this run.

    For DATED specs: list GCS by the prefix up to the ``{date}`` placeholder,
    parse the date out of each matching object's filename, and keep only
    dates within ``[today - lookback_days, today]``.

    For ROLLING specs (no placeholder): always exactly one ``(spec, None)``
    pair.
    """
    pairs: list[tuple[ArtefactSpec, date | None]] = []
    cutoff = today - timedelta(days=lookback_days)

    for spec in registry:
        if not _is_dated(spec):
            pairs.append((spec, None))
            continue

        prefix = spec.path_template.split("{date}", 1)[0]
        try:
            object_names = gcs_source.list_under(prefix, max_results=1000)
        except Exception:
            logger.exception("Failed to list GCS objects under %s", prefix)
            continue

        for obj_name in object_names:
            match = _FILENAME_DATE_PATTERN.search(obj_name)
            if match is None:
                continue
            try:
                parsed = datetime.strptime(match.group(1), "%Y%m%d").date()
            except ValueError:
                continue
            if cutoff <= parsed <= today:
                pairs.append((spec, parsed))

    logger.info(
        "Discovery: %d (spec, date) pairs across %d registry entries "
        "(cutoff=%s, today=%s)",
        len(pairs),
        len(registry),
        cutoff,
        today,
    )
    return pairs


# ----------------------------------------------------------------------------
# publish_all — main orchestration entry point
# ----------------------------------------------------------------------------


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
    """Publish every NEW (artefact, date) pair to Confluence.

    Returns:
        ``{
            "published":          [...],
            "skipped_existing":   [...],   # title already on Confluence
            "skipped_unchanged":  [...],   # rolling page with same content hash
            "errors":             [...],
            "target_date":        "2026-06-12",
            "subparent_ids":      {...},
        }``

    The ``subparent_ids`` map is exported so :func:`sweep_all` can reuse the
    page ids without re-resolving them.
    """
    if not config.paths.output_gcs_uri:
        raise ValueError(
            "paths.output_gcs_uri is not set in publisher_config.json — "
            "cannot read artefacts from GCS."
        )

    gcs_source = GCSSource(
        base_uri=config.paths.output_gcs_uri,
        bq_config=bq_config,
    )
    bq_source = BQSource(bq_config=bq_config)
    publisher = ConfluencePublisher(
        config=confluence_config,
        username=confluence_username,
        password=confluence_password,
    )

    generated_at = datetime.now(timezone.utc)
    pairs = _discover_pairs(
        registry=REGISTRY,
        gcs_source=gcs_source,
        today=target_date,
        lookback_days=confluence_config.lookback_days,
    )

    published: list[dict[str, Any]] = []
    skipped_existing: list[dict[str, Any]] = []
    skipped_unchanged: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    subparent_cache: dict[str, str] = {}

    for spec, pair_date in pairs:
        # Rate-limit defense between artefacts (matches the CI/CD-tool
        # reference's per-operation sleep pattern).
        time.sleep(random.randint(15, 20))

        if pair_date is not None:
            target = spec.path_template.format(
                date=date_to_str(pair_date), iso_date=pair_date.isoformat()
            )
            title = spec.title_template.format(
                date=date_to_str(pair_date), iso_date=pair_date.isoformat()
            )
        else:
            target = spec.path_template
            title = spec.title_template

        try:
            source = _pick_source(
                spec, gcs_source=gcs_source, bq_source=bq_source
            )

            # Dated artefacts: early title-exists skip (no GCS read needed).
            if pair_date is not None and publisher.page_exists(title=title):
                logger.info("Page exists, skipping: %s", title)
                skipped_existing.append(
                    {
                        "artefact": spec.path_template,
                        "target": target,
                        "title": title,
                    }
                )
                continue

            logger.info("Reading artefact: target=%s title=%r", target, title)
            data, raw_text_size = _read_data(spec, source, target)

            # Apply row filter BEFORE hash + render so the filtered shape is
            # what gets hashed, rendered, and attached.
            if spec.row_filter is not None and spec.render_kind == "csv":
                data = spec.row_filter(data)

            content_hash = _hash_data(spec, data)

            # Rolling artefacts: content-hash skip via state marker.
            if pair_date is None:
                state_key = _state_key_from_target(target)
                existing_marker = _load_state_marker(gcs_source, state_key)
                if (
                    existing_marker is not None
                    and existing_marker.content_hash == content_hash
                ):
                    logger.info(
                        "Rolling page unchanged, skipping: %s (hash=%s)",
                        target,
                        content_hash[:12],
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

            source_uri = (
                f"{config.paths.output_gcs_uri.rstrip('/')}/{target}"
            )
            body, item_count = _render_from_data(
                spec,
                data,
                title=title,
                source_uri=source_uri,
                generated_at=generated_at,
                row_cap=confluence_config.row_cap,
                wide_cell_columns=confluence_config.wide_cell_columns,
                run_context=run_context,
                auto_marker=confluence_config.auto_marker,
            )

            # Size gating — fall back to attachment-only placeholder if the
            # rendered body would exceed Confluence's body limit.
            initial_body_size = body_size_bytes(body)
            logger.info(
                "Rendered body for %s: %d bytes (raw source: %d bytes)",
                target,
                initial_body_size,
                raw_text_size,
            )
            if initial_body_size >= PAGE_BODY_HARD_LIMIT_BYTES:
                logger.warning(
                    "Body for %s exceeds hard limit (%d >= %d). "
                    "Falling back to attachment-only placeholder.",
                    target,
                    initial_body_size,
                    PAGE_BODY_HARD_LIMIT_BYTES,
                )
                body = render_attachment_only_placeholder(
                    title=title,
                    source_uri=source_uri,
                    generated_at=generated_at,
                    raw_size_bytes=raw_text_size,
                    attempted_render_size_bytes=initial_body_size,
                    run_context=run_context,
                    auto_marker=confluence_config.auto_marker,
                )
            elif initial_body_size >= PAGE_BODY_WARN_BYTES:
                logger.warning(
                    "Body for %s near size limit (%d bytes, warn at %d). "
                    "Page will publish but may render slowly.",
                    target,
                    initial_body_size,
                    PAGE_BODY_WARN_BYTES,
                )

            if spec.parent_title not in subparent_cache:
                subparent_cache[spec.parent_title] = (
                    publisher.ensure_subparent(title=spec.parent_title)
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

            # Persist state marker for rolling artefacts so the next run can
            # short-circuit on unchanged content.
            if pair_date is None:
                _write_state_marker(
                    gcs_source,
                    _state_key_from_target(target),
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
                    "size": item_count,
                    "body_size_bytes": body_size_bytes(body),
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
        "skipped_existing": skipped_existing,
        "skipped_unchanged": skipped_unchanged,
        "errors": errors,
        "target_date": target_date.isoformat(),
        "subparent_ids": subparent_cache,
    }


# ----------------------------------------------------------------------------
# sweep_all — retention sweep entry point
# ----------------------------------------------------------------------------


def sweep_all(
    *,
    today: date,
    confluence_config: ConfluenceConfig,
    confluence_username: str,
    confluence_password: str,
) -> dict[str, Any]:
    """Delete dated child pages older than ``confluence_config.retention_days``.

    For each unique ``parent_title`` in REGISTRY:
      1. Look up (or create) the sub-parent page id.
      2. List its children.
      3. For each child:
         - If title doesn't match the ``— YYYY-MM-DD`` pattern: keep
           (it's a rolling page like "Latest Train Metrics").
         - If parsed date is within retention window: keep.
         - If body lacks the ``auto_marker``: keep (human-owned).
         - Otherwise: delete.

    Returns a categorized summary suitable for XCom.
    """
    publisher = ConfluencePublisher(
        config=confluence_config,
        username=confluence_username,
        password=confluence_password,
    )

    cutoff = today - timedelta(days=confluence_config.retention_days)

    deleted: list[dict[str, Any]] = []
    kept_recent: list[dict[str, Any]] = []
    kept_rolling: list[dict[str, Any]] = []
    kept_human_owned: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    parent_titles_seen: set[str] = set()
    for spec in REGISTRY:
        if spec.parent_title in parent_titles_seen:
            continue
        parent_titles_seen.add(spec.parent_title)

        try:
            parent_id = publisher.ensure_subparent(title=spec.parent_title)
        except Exception as exc:
            errors.append(
                {
                    "parent_title": spec.parent_title,
                    "error": f"ensure_subparent failed: "
                    f"{type(exc).__name__}: {_sanitize(str(exc))}",
                }
            )
            continue

        try:
            children = publisher.list_children(parent_page_id=parent_id)
        except Exception as exc:
            errors.append(
                {
                    "parent_title": spec.parent_title,
                    "error": f"list_children failed: "
                    f"{type(exc).__name__}: {_sanitize(str(exc))}",
                }
            )
            continue

        for child in children:
            child_title = str(child.get("title", ""))
            child_id = str(child.get("id", ""))

            match = _TITLE_DATE_PATTERN.search(child_title)
            if match is None:
                kept_rolling.append({"title": child_title, "page_id": child_id})
                continue
            try:
                child_date = datetime.strptime(
                    match.group(1), "%Y-%m-%d"
                ).date()
            except ValueError:
                kept_rolling.append({"title": child_title, "page_id": child_id})
                continue

            if child_date >= cutoff:
                kept_recent.append(
                    {
                        "title": child_title,
                        "page_id": child_id,
                        "date": child_date.isoformat(),
                    }
                )
                continue

            # Old enough to sweep — but check the marker first so we don't
            # clobber pages a human has taken ownership of.
            if confluence_config.auto_marker:
                try:
                    body = publisher.get_page_body(page_id=child_id)
                except Exception as exc:
                    errors.append(
                        {
                            "page_id": child_id,
                            "title": child_title,
                            "error": f"get_page_body failed: "
                            f"{type(exc).__name__}: {_sanitize(str(exc))}",
                        }
                    )
                    continue
                if confluence_config.auto_marker not in body:
                    kept_human_owned.append(
                        {
                            "title": child_title,
                            "page_id": child_id,
                            "date": child_date.isoformat(),
                        }
                    )
                    logger.info(
                        "Sweep: keeping %r (human-owned, no AUTO marker).",
                        child_title,
                    )
                    continue

            try:
                publisher.delete_page(page_id=child_id)
                deleted.append(
                    {
                        "title": child_title,
                        "page_id": child_id,
                        "date": child_date.isoformat(),
                        "parent_title": spec.parent_title,
                    }
                )
                # Light delay between deletes to be polite to the API.
                time.sleep(random.randint(2, 5))
            except Exception as exc:
                errors.append(
                    {
                        "page_id": child_id,
                        "title": child_title,
                        "error": f"delete_page failed: "
                        f"{type(exc).__name__}: {_sanitize(str(exc))}",
                    }
                )

    return {
        "deleted": deleted,
        "kept_recent": kept_recent,
        "kept_rolling": kept_rolling,
        "kept_human_owned": kept_human_owned,
        "errors": errors,
        "cutoff_date": cutoff.isoformat(),
        "today": today.isoformat(),
    }


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


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


def _read_data(
    spec: ArtefactSpec,
    source: Source,
    target: str,
) -> tuple[Any, int]:
    """Read the artefact. Returns ``(data, raw_text_size_bytes)``.

    For CSV: ``data`` is a DataFrame. ``raw_text_size_bytes`` is the size of
    the canonical CSV serialization (matches what we attach + hash).

    For MD / JSON: ``data`` is the raw text string. ``raw_text_size_bytes``
    is its UTF-8 byte length.
    """
    if spec.render_kind == "csv":
        df = source.read_csv(target)
        canonical = df.to_csv(index=False)
        return df, len(canonical.encode("utf-8"))

    if spec.render_kind in ("md", "json"):
        text = source.read_text(target)
        return text, len(text.encode("utf-8"))

    raise ValueError(f"Unknown render_kind: {spec.render_kind!r}")


def _hash_data(spec: ArtefactSpec, data: Any) -> str:
    """SHA-256 of the canonical serialization. Used for content-hash skip.

    For CSV: hash the canonical CSV (``to_csv(index=False)``) — stable across
    GCS-vs-BQ source and ROW-FILTER applied vs not.

    For MD / JSON: hash the raw text.
    """
    if spec.render_kind == "csv":
        canonical = data.to_csv(index=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if spec.render_kind in ("md", "json"):
        return hashlib.sha256(data.encode("utf-8")).hexdigest()
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
    auto_marker: str,
) -> tuple[str, int]:
    """Render pre-read data. Returns ``(body_xhtml, item_count)``.

    ``item_count`` is rows for CSV and byte length for MD / JSON.
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
            auto_marker=auto_marker,
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
            auto_marker=auto_marker,
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
            auto_marker=auto_marker,
        )
        return body, len(text)

    raise ValueError(f"Unknown render_kind: {spec.render_kind!r}")


def _attachment_for(
    spec: ArtefactSpec,
    data: Any,
    target: str,
) -> tuple[bytes, str, str]:
    """Build the attachment payload for an artefact.

    Returns ``(content_bytes, content_type, display_filename)``.

    For CSV: ``df.to_csv(index=False)`` is used — this is the same form that
    :func:`_hash_data` hashes, so the attached file matches the content
    hash regardless of source (GCS text vs BQ result set) and regardless of
    whether a row filter was applied.
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
```
