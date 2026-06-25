# naming.py
#
# Single source of truth for the project-namespace identifiers used across
# this PARALLEL publisher instance. Change ``NAMESPACE`` here to rename
# consistently — every DAG id and identifier prefix derives from this
# constant.
#
# This is the SECOND publisher folder (confluence_publisher_2/), running
# side-by-side with the original Confluence_Publisher/. The DAG id is
# suffixed with ``_2`` so Airflow can load both DAGs in the same Composer
# environment without an id collision. Everything else (Source protocol,
# renderer, publisher, runtime) is identical to the original — what changes
# between the two folders is the ``publisher_config.json`` values
# (artefact, space, parent_page_id, etc.).

from __future__ import annotations

NAMESPACE: str = "jonathon"

# All DAG ids in this project should be derived from this prefix.
DAG_ID_PREFIX: str = f"{NAMESPACE}_dag"

# The publisher DAG's id. Note the ``_2`` suffix that distinguishes this
# parallel instance from the original Confluence_Publisher/ folder.
PUBLISH_DAG_ID: str = f"{DAG_ID_PREFIX}_gcs_to_confluence_publish_2"
