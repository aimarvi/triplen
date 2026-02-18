import os, time, random, logging
from urllib.parse import urlparse, parse_qs, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3, requests
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig

LOG = logging.getLogger('scidb_to_s3')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ============ SOME CONFIG PARAMS ===============
URLS_TXT_PATH = '../../test_url.txt'  # set to your file path
S3_BUCKET = 'visionlab-members'
S3_PREFIX = f'amarvi/datasets'  # customize
MAX_WORKERS = 6
SKIP_IF_EXISTS = True

# For large files: multipart settings
TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=64 * 1024 * 1024,  # 64 MiB
    multipart_chunksize=64 * 1024 * 1024,
    max_concurrency=4,
    use_threads=True,
)

# HTTP settings
CHUNK_TIMEOUT = (10, 300)  # (connect, read)
USER_AGENT = 'scidb-s3-streamer/1.0'


def load_urls(txt_path: str) -> list[str]:
    urls: list[str] = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
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
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def stream_http_to_s3(url: str, bucket: str, key: str, *, max_retries: int = 8) -> None:
    """
    Streams ScienceDB HTTP download directly into S3 object `s3://bucket/key`.
    Retries on common transient issues (429/5xx/timeouts).
    """
    s3 = boto3.client("s3")

    if SKIP_IF_EXISTS and s3_object_exists(s3, bucket, key):
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
                boto3.client("s3").upload_fileobj(r.raw, bucket, key, Config=TRANSFER_CONFIG)

            LOG.info("OK   %s -> s3://%s/%s", url, bucket, key)
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

if __name__== '__main__':
    print('Testing...')

    urls = load_urls(URLS_TXT_PATH)
    LOG.info("Loaded %d URLs", len(urls))

    # Precompute keys for nicer logging / debugging
    work = [(u, derive_s3_key(u, S3_PREFIX)) for u in urls]
    print(work[0])

    for (u, k) in work:
        stream_http_to_s3(u, S3_BUCKET, k)
