"""JSON read/write helpers and GCS sync for Vertex AI custom training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {uri!r}")
    rest = uri[5:]
    bucket, _, blob = rest.partition("/")
    if not bucket or not blob:
        raise ValueError(f"Invalid GCS URI: {uri!r}")
    return bucket, blob.rstrip("/")


def download_gcs_file(gcs_uri: str, local_path: Path) -> None:
    """Download a single GCS object to ``local_path``."""
    from google.cloud import storage

    bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(
        str(local_path)
    )


def upload_gcs_file(local_path: Path, gcs_uri: str) -> None:
    """Upload a local file to the GCS object named by ``gcs_uri``."""
    from google.cloud import storage

    bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
    storage.Client().bucket(bucket_name).blob(blob_name).upload_from_filename(
        str(local_path)
    )


def sync_gcs_prefix_to_dir(gcs_prefix_uri: str, local_dir: Path) -> None:
    """Download all objects under a ``gs://bucket/prefix/`` into ``local_dir``."""
    from google.cloud import storage

    prefix_uri = gcs_prefix_uri.rstrip("/") + "/"
    bucket_name, prefix = _parse_gcs_uri(prefix_uri)
    local_dir.mkdir(parents=True, exist_ok=True)
    for blob in storage.Client().list_blobs(bucket_name, prefix=prefix + "/"):
        if blob.name.endswith("/"):
            continue
        rel = blob.name[len(prefix) + 1 :]
        if not rel:
            continue
        dest = local_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))


def save_json(obj: Any, path: Path) -> None:
    """Write ``obj`` to ``path`` as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


def load_json(path: Path) -> Any:
    """Read a JSON file and return the parsed object."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
