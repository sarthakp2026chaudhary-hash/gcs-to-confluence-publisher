# confluence_publisher.py

"""Confluence client wrapper.

Title-based create-or-update against the configured space. Every API call is
wrapped with :func:`_with_retry` — 5 attempts with exponential backoff
(10, 20, 30, 40 seconds between attempts). Matches the retry pattern used by
the CI/CD-tool reference (``refresh_confl_recursion``); rationale is that
Confluence APIs occasionally rate-limit or return transient 5xx and a single
attempt fails the whole DAG run unnecessarily.

Errors that exhaust the retry budget surface as :class:`ConfluencePublishError`
with a sanitized message naming the connection id (never the credential).

The ``atlassian`` import is deferred to the constructor so the DAG file is
still parseable by the scheduler when ``atlassian-python-api`` has not yet
been added via Composer → Environment → PyPI packages.

``verify_ssl=False`` is supported via :attr:`ConfluenceConfig.verify_ssl` for
internal Server / Data Center hosts whose corporate cert chain isn't in
Python's default trust store.
"""

from __future__ import annotations

import html
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Callable, TypeVar

from config import ConfluenceConfig

logger = logging.getLogger(__name__)


class ConfluencePublishError(RuntimeError):
    """Raised when a Confluence API call fails after retries.

    The exception message names the connection id (not the credential value)
    and includes a sanitized version of the underlying error.
    """


_BASIC_AUTH_PATTERN = re.compile(r"(Basic\s+)[A-Za-z0-9+/=]+", re.IGNORECASE)
_QUERY_AUTH_PATTERN = re.compile(
    r"([?&](?:password|os_password|auth_token|api_token|api-token)=)[^&\s]+",
    re.IGNORECASE,
)

# Retry policy: 5 attempts with exponential backoff. Sleeps before retry are
# 10, 20, 30, 40 seconds — total worst-case wait ~100s before giving up. Same
# shape as the CI/CD-tool reference (MAX_RETRIES=5, DELAY*attempt backoff).
_MAX_RETRIES = 5
_RETRY_BASE_DELAY_SECONDS = 10


T = TypeVar("T")


def _sanitize(message: str) -> str:
    """Redact credential-looking substrings from an error message."""
    message = _BASIC_AUTH_PATTERN.sub(r"\1<redacted>", message)
    message = _QUERY_AUTH_PATTERN.sub(r"\1<redacted>", message)
    return message


def _with_retry(call: Callable[[], T], *, op_name: str) -> T:
    """Invoke ``call()`` with retry + exponential backoff.

    Retries on any exception up to :data:`_MAX_RETRIES` times. Sleeps
    ``_RETRY_BASE_DELAY_SECONDS * attempt`` seconds between attempts. The
    last exception is re-raised if all attempts fail; the caller wraps it as
    a :class:`ConfluencePublishError` with full surrounding context.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return call()
        except Exception as exc:
            last_exc = exc
            backoff = _RETRY_BASE_DELAY_SECONDS * attempt
            if attempt < _MAX_RETRIES:
                logger.warning(
                    "%s attempt %s/%s failed (%s) — sleeping %ss before retry.",
                    op_name,
                    attempt,
                    _MAX_RETRIES,
                    type(exc).__name__,
                    backoff,
                )
                time.sleep(backoff)
            else:
                logger.error(
                    "%s exhausted all %s retries — giving up.",
                    op_name,
                    _MAX_RETRIES,
                )
    assert last_exc is not None
    raise last_exc


class ConfluencePublisher:
    """Title-based create-or-update Confluence publisher with retries."""

    def __init__(
        self,
        *,
        config: ConfluenceConfig,
        username: str,
        password: str,
    ) -> None:
        from atlassian import Confluence  # deferred: PyPI-installed in Composer

        if not username or not password:
            raise ValueError(
                "Confluence credentials missing — username and password must both be set "
                f"(connection_id was {config.auth_connection_id!r})."
            )

        self._config = config
        self._client = Confluence(
            url=config.base_url,
            username=username,
            password=password,
            cloud=(config.flavor == "cloud"),
            verify_ssl=config.verify_ssl,
        )

    def probe(self) -> bool:
        """Lightweight reachability check — returns True if the space is visible.

        Used by ``check_dependencies`` in the DAG to fail fast on bad creds or
        a wrong base URL, before any artefact is touched.
        """
        try:
            _with_retry(
                lambda: self._client.get_space(self._config.space_key),
                op_name="get_space",
            )
            return True
        except Exception as exc:
            raise ConfluencePublishError(
                f"Confluence reachability probe failed after {_MAX_RETRIES} retries for "
                f"connection_id={self._config.auth_connection_id!r} "
                f"space={self._config.space_key!r}: "
                f"{type(exc).__name__}: {_sanitize(str(exc))}"
            ) from exc

    def create_or_update(
        self,
        *,
        title: str,
        body_xhtml: str,
        parent_page_id: str | None = None,
    ) -> str:
        """Create the page or update it if it already exists. Returns the page id.

        Lookup is by exact title within the configured space. If found, the page
        is updated and Confluence bumps its version. If not found, a new child
        of ``parent_page_id`` (or the config default) is created. All three API
        calls (get_page_by_title, update_page, create_page) retry on failure.
        """
        parent_id = parent_page_id or self._config.parent_page_id
        space = self._config.space_key

        try:
            existing = _with_retry(
                lambda: self._client.get_page_by_title(space=space, title=title),
                op_name=f"get_page_by_title({title!r})",
            )
        except Exception as exc:
            raise ConfluencePublishError(
                f"Confluence get_page_by_title failed after {_MAX_RETRIES} retries for title={title!r} "
                f"connection_id={self._config.auth_connection_id!r}: "
                f"{type(exc).__name__}: {_sanitize(str(exc))}"
            ) from exc

        if existing:
            page_id = str(existing["id"])
            logger.info("Updating Confluence page: id=%s title=%r", page_id, title)
            try:
                _with_retry(
                    lambda: self._client.update_page(
                        page_id=page_id,
                        title=title,
                        body=body_xhtml,
                        representation="storage",
                    ),
                    op_name=f"update_page(page_id={page_id})",
                )
            except Exception as exc:
                raise ConfluencePublishError(
                    f"Confluence update_page failed after {_MAX_RETRIES} retries for "
                    f"page_id={page_id} title={title!r} "
                    f"connection_id={self._config.auth_connection_id!r}: "
                    f"{type(exc).__name__}: {_sanitize(str(exc))}"
                ) from exc
            return page_id

        logger.info(
            "Creating Confluence page: title=%r parent_id=%s space=%s",
            title,
            parent_id,
            space,
        )
        try:
            created = _with_retry(
                lambda: self._client.create_page(
                    space=space,
                    title=title,
                    body=body_xhtml,
                    parent_id=parent_id,
                    representation="storage",
                ),
                op_name=f"create_page({title!r})",
            )
        except Exception as exc:
            raise ConfluencePublishError(
                f"Confluence create_page failed after {_MAX_RETRIES} retries for title={title!r} "
                f"parent_id={parent_id} connection_id={self._config.auth_connection_id!r}: "
                f"{type(exc).__name__}: {_sanitize(str(exc))}"
            ) from exc
        return str(created["id"])

    def ensure_subparent(self, *, title: str) -> str:
        """Return the page id of the sub-parent with this title.

        If a page with this title exists in the space, return its id. Otherwise
        create it as a child of the root ``parent_page_id`` from config and
        return the new id. Both API calls retry on failure.
        """
        space = self._config.space_key

        try:
            existing = _with_retry(
                lambda: self._client.get_page_by_title(space=space, title=title),
                op_name=f"get_page_by_title(subparent={title!r})",
            )
        except Exception as exc:
            raise ConfluencePublishError(
                f"Confluence get_page_by_title (sub-parent) failed after {_MAX_RETRIES} retries "
                f"for title={title!r} "
                f"connection_id={self._config.auth_connection_id!r}: "
                f"{type(exc).__name__}: {_sanitize(str(exc))}"
            ) from exc

        if existing:
            return str(existing["id"])

        logger.info(
            "Creating Confluence sub-parent: title=%r under parent_id=%s",
            title,
            self._config.parent_page_id,
        )
        body = (
            f"<p>Auto-created index for <strong>{html.escape(title)}</strong>. "
            "Dated artefacts publish as children of this page.</p>"
        )
        try:
            created = _with_retry(
                lambda: self._client.create_page(
                    space=space,
                    title=title,
                    body=body,
                    parent_id=self._config.parent_page_id,
                    representation="storage",
                ),
                op_name=f"create_page(subparent={title!r})",
            )
        except Exception as exc:
            raise ConfluencePublishError(
                f"Confluence create_page (sub-parent) failed after {_MAX_RETRIES} retries "
                f"for title={title!r} "
                f"connection_id={self._config.auth_connection_id!r}: "
                f"{type(exc).__name__}: {_sanitize(str(exc))}"
            ) from exc
        return str(created["id"])

    def attach_source(
        self,
        *,
        page_id: str,
        filename: str,
        content: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        """Attach raw source bytes to an existing Confluence page.

        The page already renders the data as a human-readable XHTML table /
        formatted block. The attachment is the structured payload — CSV / JSON
        / Markdown — that an agent consumer can download via the Confluence
        REST API and parse natively, without scraping ``<td>`` tags.

        ``filename`` is the display name in Confluence (and the basename the
        agent downloads under). Bytes are written to a tempfile and uploaded
        via ``atlassian.Confluence.attach_file`` (retried on failure); the
        tempfile is removed in a ``finally`` block regardless of success.
        """
        suffix = Path(filename).suffix or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            _with_retry(
                lambda: self._client.attach_file(
                    filename=tmp_path,
                    name=filename,
                    content_type=content_type,
                    page_id=page_id,
                ),
                op_name=f"attach_file({filename!r})",
            )
            logger.info(
                "Attached source to Confluence page: page_id=%s filename=%r content_type=%s bytes=%s",
                page_id,
                filename,
                content_type,
                len(content),
            )
        except Exception as exc:
            raise ConfluencePublishError(
                f"Confluence attach_file failed after {_MAX_RETRIES} retries for filename={filename!r} "
                f"page_id={page_id} connection_id={self._config.auth_connection_id!r}: "
                f"{type(exc).__name__}: {_sanitize(str(exc))}"
            ) from exc
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
