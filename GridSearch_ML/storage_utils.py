# storage_utils.py

from __future__ import annotations

import io
import logging
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from google.cloud import storage

logger = logging.getLogger(__name__)


class GCSOutputStore:

    def __init__(self, base_uri: str):
        parsed = urlparse(base_uri)
        if parsed.scheme != "gs":
            raise ValueError(f"Invalid GCS URI: {base_uri}")
        self.base_uri = base_uri.rstrip("/")
        self.bucket_name = parsed.netloc
        self.base_prefix = parsed.path.lstrip("/").rstrip("/")
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)

    def upload_file(self, local_path: Path, remote_name: str | None = None) -> str:
        remote_name = remote_name or local_path.name
        object_name = f"{self.base_prefix}/{remote_name}" if self.base_prefix else remote_name
        blob = self.bucket.blob(object_name)
        blob.upload_from_filename(str(local_path))
        gcs_uri = f"gs://{self.bucket_name}/{object_name}"
        logger.info("Uploaded %s to %s", local_path, gcs_uri)
        return gcs_uri

    def download_text(self, remote_name: str) -> str:
        object_name = f"{self.base_prefix}/{remote_name}" if self.base_prefix else remote_name
        blob = self.bucket.blob(object_name)
        if not blob.exists():
            raise FileNotFoundError(f"GCS object not found: gs://{self.bucket_name}/{object_name}")
        return blob.download_as_text(encoding="utf-8")

    def download_file(self, remote_name: str, local_path: Path) -> Path:
        object_name = f"{self.base_prefix}/{remote_name}" if self.base_prefix else remote_name
        blob = self.bucket.blob(object_name)
        if not blob.exists():
            raise FileNotFoundError(f"GCS object not found: gs://{self.bucket_name}/{object_name}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        logger.info("Downloaded gs://%s/%s to %s", self.bucket_name, object_name, local_path)
        return local_path

    def download_csv(self, remote_name: str) -> pd.DataFrame:
        text = self.download_text(remote_name)
        return pd.read_csv(io.StringIO(text))
