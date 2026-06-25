# confluence_renderer.py

"""Render artefacts as Confluence storage-format XHTML.

Three renderers (one per artefact ``render_kind`` in the runtime registry):
    :func:`render_csv_table`  — DataFrame → table with wide-cell wrap.
    :func:`render_markdown`   — markdown text → XHTML (uses ``markdown``
                                package if available, falls back to escaped
                                ``<pre>`` so a missing dep does not abort).
    :func:`render_json_table` — JSON-like object → recursive key/value table.

All three share :func:`_render_footer` for the provenance block at the bottom
of every page.
"""

from __future__ import annotations

import html
import json
import logging
from datetime import datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


_WIDE_CELL_STYLE = (
    "max-width:60ch; word-break:break-word; "
    "font-family:monospace; font-size:smaller;"
)


def render_csv_table(
    df: pd.DataFrame,
    *,
    title: str,
    source_uri: str,
    generated_at: datetime,
    row_cap: int = 5000,
    wide_cell_columns: tuple[str, ...] = (),
    run_context: dict[str, str] | None = None,
) -> str:
    """Return the Confluence storage-format XHTML body for a DataFrame.

    Cells in ``wide_cell_columns`` are wrapped in a styled ``<div>`` and their
    ``\\n`` are converted to ``<br/>`` so stack-trace excerpts wrap rather than
    overflow. Columns named here but absent from ``df`` are silently ignored —
    the union list in config can cover many artefact types.
    """
    rows_total = len(df)
    truncated = rows_total > row_cap
    if truncated:
        logger.warning(
            "Row cap hit for %r: %s rows of %s — truncating.",
            title,
            row_cap,
            rows_total,
        )
        df = df.head(row_cap)

    columns = [str(col) for col in df.columns]
    wide_set = set(wide_cell_columns)

    header_cells = "".join(f"<th>{html.escape(col)}</th>" for col in columns)

    body_rows: list[str] = []
    for _, row in df.iterrows():
        cells: list[str] = []
        for col in columns:
            escaped = html.escape(_cell_to_str(row[col]))
            if col in wide_set:
                escaped = escaped.replace("\n", "<br/>")
                cells.append(
                    f'<td><div style="{_WIDE_CELL_STYLE}">{escaped}</div></td>'
                )
            else:
                cells.append(f"<td>{escaped}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    body = "".join(body_rows)

    table_xhtml = (
        f"<table><thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )

    title_xhtml = f"<h1>{html.escape(title)}</h1>"

    truncation_note = ""
    if truncated:
        truncation_note = (
            f"<p><strong>Truncated at first {row_cap} rows of "
            f"{rows_total} total.</strong></p>"
        )

    footer = _render_footer(
        source_uri=source_uri,
        generated_at=generated_at,
        rows_total=rows_total,
        run_context=run_context,
    )

    return title_xhtml + truncation_note + table_xhtml + footer


def render_markdown(
    text: str,
    *,
    title: str,
    source_uri: str,
    generated_at: datetime,
    run_context: dict[str, str] | None = None,
) -> str:
    """Render markdown text as a Confluence page body.

    Uses the ``markdown`` package when available; otherwise falls back to an
    escaped ``<pre>`` block so a missing dep produces a readable (if uglier)
    page rather than aborting the publish.
    """
    title_xhtml = f"<h1>{html.escape(title)}</h1>"
    body_html = _markdown_to_xhtml(text)
    footer = _render_footer(
        source_uri=source_uri,
        generated_at=generated_at,
        rows_total=None,
        run_context=run_context,
    )
    return title_xhtml + body_html + footer


def render_json_table(
    obj: Any,
    *,
    title: str,
    source_uri: str,
    generated_at: datetime,
    run_context: dict[str, str] | None = None,
    max_depth: int = 2,
) -> str:
    """Render a JSON-like object as a Confluence key/value table.

    Dicts become two-column tables (key | value). Lists become ``<ul>``s.
    Recursion stops at ``max_depth`` — anything deeper renders as a
    ``<code>`` JSON literal so the page never explodes.
    """
    title_xhtml = f"<h1>{html.escape(title)}</h1>"
    body_html = _json_to_xhtml(obj, depth=0, max_depth=max_depth)
    footer = _render_footer(
        source_uri=source_uri,
        generated_at=generated_at,
        rows_total=None,
        run_context=run_context,
    )
    return title_xhtml + body_html + footer


def _render_footer(
    *,
    source_uri: str,
    generated_at: datetime,
    rows_total: int | None,
    run_context: dict[str, str] | None,
) -> str:
    lines = [
        f"Source: <code>{html.escape(source_uri)}</code>",
        f"Generated: {html.escape(generated_at.isoformat())}",
    ]
    if rows_total is not None:
        lines.append(f"Rows: {rows_total}")
    if run_context:
        for key in ("dag_run_id", "logical_date"):
            value = run_context.get(key)
            if value:
                lines.append(
                    f"{html.escape(key)}: <code>{html.escape(str(value))}</code>"
                )
    return "<hr/><p><em>" + "<br/>".join(lines) + "</em></p>"


def _markdown_to_xhtml(text: str) -> str:
    try:
        import markdown as _md  # deferred: optional Composer PyPI dep

        return _md.markdown(text, extensions=["fenced_code", "tables"])
    except ImportError:
        logger.warning(
            "markdown package not installed — rendering as escaped <pre>. "
            "Install via Composer → Environment → PyPI packages for nicer output."
        )
        return (
            "<p><em>markdown package unavailable — rendering as preformatted text.</em></p>"
            f"<pre>{html.escape(text)}</pre>"
        )


def _json_to_xhtml(obj: Any, *, depth: int, max_depth: int) -> str:
    if isinstance(obj, dict):
        if depth >= max_depth:
            return f"<code>{html.escape(json.dumps(obj))}</code>"
        rows: list[str] = []
        for key, value in obj.items():
            value_html = _json_to_xhtml(value, depth=depth + 1, max_depth=max_depth)
            rows.append(
                f"<tr><th>{html.escape(str(key))}</th><td>{value_html}</td></tr>"
            )
        return f"<table>{''.join(rows)}</table>"

    if isinstance(obj, list):
        if depth >= max_depth:
            return f"<code>{html.escape(json.dumps(obj))}</code>"
        items = [
            f"<li>{_json_to_xhtml(item, depth=depth + 1, max_depth=max_depth)}</li>"
            for item in obj
        ]
        return f"<ul>{''.join(items)}</ul>"

    if obj is None:
        return "<em>null</em>"
    if isinstance(obj, bool):
        return "<code>true</code>" if obj else "<code>false</code>"
    if isinstance(obj, (int, float)):
        return f"<code>{html.escape(str(obj))}</code>"
    return html.escape(str(obj))


def _cell_to_str(value: object) -> str:
    """Stringify a DataFrame cell. NaN/None render as empty strings."""
    if pd.isna(value):
        return ""
    return str(value)
