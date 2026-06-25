# `requirements.txt`

Source path: `confluence_publisher_2/requirements.txt`

```
# requirements.txt — Python 3.11.8
#
# For LOCAL development of the Confluence_Publisher DAG (venv, IDE import
# resolution, unit tests, dry runs against mock data).
#
# In Cloud Composer most of these are already in the image. Only ADD via
# Composer → Environment → PyPI packages:
#   - atlassian-python-api
#   - markdown               (optional; render_markdown falls back to <pre>)
#
# Everything else below is also in the Composer image — listed here so a
# local venv mirrors what runs in production.

# ============================================================================
# Confluence + Atlassian client (REQUIRED for the publisher)
# ============================================================================
atlassian-python-api>=3.41.0,<4.0
requests>=2.31.0                  # transitive dep of atlassian, pinned for SSL/TLS predictability
urllib3>=1.26.18,<3.0             # transitive; verify_ssl=False warnings come from this layer

# ============================================================================
# Data manipulation
# ============================================================================
pandas>=2.1.0,<3.0
numpy>=1.24.0,<2.0
pyarrow>=14.0.0                   # parquet support via pandas, used by GridSearch_ML

# ============================================================================
# Google Cloud SDKs
# ============================================================================
google-cloud-storage>=2.10.0
google-cloud-bigquery>=3.13.0
google-auth>=2.23.0
db-dtypes>=1.2.0                  # REQUIRED by bigquery.to_dataframe() — without this, Phase E fails

# ============================================================================
# Markdown rendering (OPTIONAL — renderer falls back to escaped <pre>)
# ============================================================================
markdown>=3.5

# ============================================================================
# Apache Airflow — see note below; DO NOT install via this file
# ============================================================================
# Airflow has 600+ transitive deps and unbounded version conflicts. Always
# install with the official constraints file:
#
#   pip install "apache-airflow==2.9.0" \
#       --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.0/constraints-3.11.txt"
#   pip install "apache-airflow-providers-google>=10.10.0"
#
# In Composer, Airflow is already installed and version-pinned by the image.
# Local install is only needed if you want to import-resolve the DAG file
# outside Composer (IDE, mypy, unit tests against the DAG).

# ============================================================================
# Standard library imports (no PyPI deps needed — listed for reference):
#   json, hashlib, logging, random, time, html, io, re, os, tempfile, sys
#   dataclasses, datetime, pathlib, typing, urllib.parse
# ============================================================================
```
