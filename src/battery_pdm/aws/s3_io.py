"""Transparent S3 + local path I/O.

The flows are written with file paths like 'outputs/alarms.parquet'. In AWS we
pass 's3://battery-pdm-dev-data-998716768706/alarms.parquet' instead. These
helpers let the same code work in both contexts without if-branches everywhere.

Usage:
    from battery_pdm.aws.s3_io import read_parquet, write_parquet
    df = read_parquet("s3://bucket/key.parquet")   # works
    df = read_parquet("outputs/local.parquet")     # also works
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd


def _is_s3(path: str) -> bool:
    return str(path).startswith("s3://")


def _split_s3(uri: str) -> tuple[str, str]:
    rest = uri[len("s3://") :]
    bucket, _, key = rest.partition("/")
    return bucket, key


def read_parquet(path: str) -> pd.DataFrame:
    """Read parquet from S3 or local. Returns empty DataFrame if S3 key missing."""
    if not _is_s3(path):
        return pd.read_parquet(path)
    import boto3
    from botocore.exceptions import ClientError

    bucket, key = _split_s3(path)
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return pd.DataFrame()
        raise


def write_parquet(df: pd.DataFrame, path: str) -> None:
    """Write parquet to S3 or local. Creates parent dirs as needed."""
    if not _is_s3(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return
    import boto3

    bucket, key = _split_s3(path)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


def read_json(path: str) -> dict:
    import json

    if not _is_s3(path):
        return json.loads(Path(path).read_text())
    import boto3

    bucket, key = _split_s3(path)
    obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())


def write_json(data: dict, path: str) -> None:
    import json

    if not _is_s3(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2))
        return
    import boto3

    bucket, key = _split_s3(path)
    boto3.client("s3").put_object(
        Bucket=bucket, Key=key, Body=json.dumps(data, indent=2).encode()
    )


def exists(path: str) -> bool:
    if not _is_s3(path):
        return Path(path).exists()
    import boto3
    from botocore.exceptions import ClientError

    bucket, key = _split_s3(path)
    try:
        boto3.client("s3").head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def join_path(base: str, *parts: str) -> str:
    """Join path segments. S3-aware: preserves s3:// prefix."""
    if _is_s3(base):
        combined = base.rstrip("/") + "/" + "/".join(p.strip("/") for p in parts)
        return combined
    return str(Path(base).joinpath(*parts))


def read_bytes(path: str) -> bytes:
    if not _is_s3(path):
        return Path(path).read_bytes()
    import boto3

    bucket, key = _split_s3(path)
    obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def write_bytes(path: str, data: bytes) -> None:
    if not _is_s3(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)
        return
    import boto3

    bucket, key = _split_s3(path)
    boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=data)


def write_text(path: str, text: str) -> None:
    if not _is_s3(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
        return
    import boto3

    bucket, key = _split_s3(path)
    boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=text.encode())


def read_text(path: str) -> str:
    if not _is_s3(path):
        return Path(path).read_text()
    import boto3

    bucket, key = _split_s3(path)
    obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode()


def mkdir(path: str) -> None:
    """Create directory (no-op for S3)."""
    if not _is_s3(path):
        Path(path).mkdir(parents=True, exist_ok=True)
