# `naming.py`

Source path: `Confluence_Publisher/naming.py`

```python
# naming.py
#
# Single source of truth for the project-namespace identifiers used across
# the publisher. Change ``NAMESPACE`` here to rename consistently — every
# DAG id and identifier prefix derives from this constant.
#
# Rationale: keeps the rename surface to ONE line. No find-and-replace across
# multiple files. New identifiers (additional DAG ids, page-title prefixes,
# etc.) should be added here as derived constants rather than hardcoded
# elsewhere.

from __future__ import annotations

NAMESPACE: str = "jonathon"

# All DAG ids in this project should be derived from this prefix.
DAG_ID_PREFIX: str = f"{NAMESPACE}_dag"

# The publisher DAG's id. Imported by airflow_dag_confluence.py.
PUBLISH_DAG_ID: str = f"{DAG_ID_PREFIX}_gcs_to_confluence_publish"
```
