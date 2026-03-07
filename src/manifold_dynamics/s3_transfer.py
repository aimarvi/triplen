from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import boto3
import requests
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig

LOG = logging.getLogger("scidb_to_s3")

# Transfer configuration
URLS_TXT_PATH = Path("./../../scidb_url.txt")
S3_BUCKET = "visionlab-members"
S3_PREFIX = "amarvi/datasets/triple-n/"
MAX_WORKERS = 6
SKIP_IF_EXISTS = True

# Multipart upload settings for large files
TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=64 * 1024 * 1024,  # 64 MiB
    multipart_chunksize=64 * 1024 * 1024,
    max_concurrency=4,
    use_threads=True,
)

# HTTP request settings
CHUNK_TIMEOUT = (10, 300)  # (connect, read)
USER_AGENT = "scidb-s3-streamer/1.0"


def load_urls(txt_path: Path | str) -> list[str]:
    """Load transfer URLs from a newline-delimited text file."""
    urls: list[str] = []
    with Path(txt_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    return urls


def derive_s3_key(url: str, prefix: str) -> str:
    """
    Prefer keeping the ScienceDB `path` structure, e.g.
      ...&path=/V1/Raw/.../file.mat -> <prefix>/V1/Raw/.../file.mat

    Fallback to `fileName` if `path` is missing.
    """
    q = parse_qs(urlparse(url).query)

    if "path" in q and q["path"]:
        path = unquote(q["path"][0])
        path = path.lstrip("/")  # avoid absolute-style keys
        return f"{prefix.rstrip('/')}/{path}"

    if "fileName" in q and q["fileName"]:
        fname = unquote(q["fileName"][0])
        return f"{prefix.rstrip('/')}/{fname}"

    # Last resort: use fileId
    file_id = q.get("fileId", ["unknown"])[0]
    return f"{prefix.rstrip('/')}/{file_id}"


def s3_object_exists(s3_client, bucket: str, key: str) -> bool:
    """Return True when an S3 object exists, False when it does not."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def stream_http_to_s3(
    s3_client,
    url: str,
    bucket: str,
    key: str,
    *,
    max_retries: int = 8,
) -> None:
    """
    Streams ScienceDB HTTP download directly into S3 object `s3://bucket/key`.
    Retries on common transient issues (429/5xx/timeouts).
    """
    if SKIP_IF_EXISTS and s3_object_exists(s3_client, bucket, key):
        LOG.info("SKIP exists s3://%s/%s", bucket, key)
        return

    # Basic exponential backoff with jitter
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(
                url,
                stream=True,
                allow_redirects=True,
                timeout=CHUNK_TIMEOUT,
                headers={"User-Agent": USER_AGENT},
            ) as r:
                # Retryable status codes
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"retryable HTTP {r.status_code}", response=r)

                r.raise_for_status()

                # Upload by streaming from response raw socket/file
                s3_client.upload_fileobj(r.raw, bucket, key, Config=TRANSFER_CONFIG)

            LOG.info("✓ %s", key)
            return

        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            # Decide whether retry
            status = getattr(getattr(e, "response", None), "status_code", None)
            retryable = (status in (429, 500, 502, 503, 504)) or isinstance(
                e, (requests.Timeout, requests.ConnectionError)
            )

            if not retryable or attempt == max_retries:
                LOG.error("FAIL %s -> s3://%s/%s (%s)", url, bucket, key, e)
                raise

            sleep_s = min(60.0, (2 ** (attempt - 1))) + random.random()
            LOG.warning(
                "RETRY %d/%d in %.1fs for %s (status=%s, err=%s)",
                attempt, max_retries, sleep_s, url, status, e
            )
            time.sleep(sleep_s)


def run_transfer(urls_txt_path: Path | str = URLS_TXT_PATH) -> None:
    """Run URL-to-S3 streaming transfer for all URLs in the input file."""
    urls = load_urls(urls_txt_path)
    LOG.info("Loaded %d URL(s)", len(urls))

    work = [(u, derive_s3_key(u, S3_PREFIX)) for u in urls]
    if work:
        LOG.info("First target key: s3://%s/%s", S3_BUCKET, work[0][1])

    s3_client = boto3.client("s3")
    for url, key in work:
        stream_http_to_s3(s3_client, url, S3_BUCKET, key)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_transfer(URLS_TXT_PATH)


if __name__ == "__main__":
    main()
