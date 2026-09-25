"""Fetch FMA metadata and only the audio clips needed for training.

Usage (from backend/):
    python -m training.download metadata      # fma_metadata.zip -> data/fma_metadata/*.csv
    python -m training.download targets       # -> data/targets.parquet
    python -m training.download audio         # needed clips from fma_large.zip -> data/fma_large/

The audio step uses HTTP range requests: it reads the zip's central directory once, then
fetches each wanted member's bytes directly, so the 93 GiB archive is never downloaded.
It is resumable (existing verified clips are skipped) and verifies each clip's CRC-32.
"""

import argparse
import bz2
import hashlib
import shutil
import struct
import sys
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from remotezip import RemoteZip

from training.common import AUDIO_DIR, DATA_DIR, METADATA_DIR, TARGETS, TARGETS_PATH, audio_path, human_bytes

METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
METADATA_SHA1 = "f0df49ffe5f2a6008d7dc83c6915b31835dfe733"
LARGE_URL = "https://os.unil.cloud.switch.ch/fma/fma_large.zip"
METADATA_FILES = ["tracks.csv", "echonest.csv", "features.csv", "genres.csv"]

AUDIO_LOG = DATA_DIR / "download_audio.log"
_LOCAL_HEADER = struct.Struct("<4sHHHHHIIIHH")  # zip local file header, 30 bytes


class PermanentError(RuntimeError):
    """A failure that retrying cannot fix (e.g. a corrupt archive member)."""


# --- Metadata ---------------------------------------------------------------------------


def _sha1(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_metadata() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "fma_metadata.zip"
    part = zip_path.with_suffix(".zip.part")

    if not zip_path.exists():
        done = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={done}-"} if done else {}
        with requests.get(METADATA_URL, headers=headers, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = done + int(r.headers.get("Content-Length", 0))
            mode = "ab" if r.status_code == 206 else "wb"
            done = done if r.status_code == 206 else 0
            start, last = time.time(), 0.0
            with open(part, mode) as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if time.time() - last > 5:
                        last = time.time()
                        rate = done / max(last - start, 1e-9)
                        print(f"  {human_bytes(done)} / {human_bytes(total)} ({human_bytes(rate)}/s)", flush=True)
        part.rename(zip_path)

    print("Verifying SHA-1 ...")
    digest = _sha1(zip_path)
    if digest != METADATA_SHA1:
        zip_path.unlink()
        sys.exit(f"SHA-1 mismatch for fma_metadata.zip ({digest}); deleted it, please re-run.")
    print(f"  ok ({digest})")

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for name in METADATA_FILES:
            with zf.open(f"fma_metadata/{name}") as src, open(METADATA_DIR / name, "wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"  extracted {name} ({human_bytes((METADATA_DIR / name).stat().st_size)})")


# --- Targets ----------------------------------------------------------------------------


def build_targets() -> pd.DataFrame:
    """One row per track with Echo Nest audio features: targets, tempo, artist, duration."""
    echonest = pd.read_csv(METADATA_DIR / "echonest.csv", index_col=0, header=[0, 1, 2])
    audio_features = echonest["echonest", "audio_features"]
    tracks = pd.read_csv(METADATA_DIR / "tracks.csv", index_col=0, header=[0, 1])

    df = audio_features[TARGETS + ["tempo"]].rename(columns={"tempo": "echonest_tempo"})
    df = df.join(
        tracks[[("artist", "id"), ("track", "duration"), ("set", "subset")]].set_axis(
            ["artist_id", "duration_s", "fma_subset"], axis=1
        ),
        how="inner",
    )
    df.index.name = "track_id"
    df = df.dropna(subset=TARGETS + ["artist_id"])
    df["artist_id"] = df["artist_id"].astype(int)
    df.to_parquet(TARGETS_PATH)

    print(f"{len(df)} tracks with Echo Nest audio features, {df['artist_id'].nunique()} artists")
    print(f"FMA subsets: {df['fma_subset'].value_counts().to_dict()}")
    print(df[TARGETS + ["echonest_tempo"]].describe().round(3).to_string())
    return df


# --- Audio ------------------------------------------------------------------------------


def _fetch_member(session: requests.Session, info: zipfile.ZipInfo) -> bytes:
    """Fetch one zip member's bytes with a single range request and verify its CRC-32."""
    # Local header + name + extra field + data. The local extra field can differ in length
    # from the central directory's, so over-fetch a little and parse the local header.
    start = info.header_offset
    end = start + _LOCAL_HEADER.size + len(info.filename.encode()) + info.compress_size + 1024
    r = session.get(LARGE_URL, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=120)
    r.raise_for_status()
    if r.status_code != 206:
        raise RuntimeError(f"server ignored the range request (HTTP {r.status_code})")
    blob = r.content
    fields = _LOCAL_HEADER.unpack_from(blob)
    if fields[0] != b"PK\x03\x04":
        raise RuntimeError("bad local file header")
    data_start = _LOCAL_HEADER.size + fields[9] + fields[10]
    raw = blob[data_start : data_start + info.compress_size]
    if len(raw) != info.compress_size:
        raise RuntimeError("truncated member data")
    if info.compress_type == zipfile.ZIP_STORED:
        data = raw
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(raw, -15)
    elif info.compress_type == zipfile.ZIP_BZIP2:  # what fma_large.zip uses
        data = bz2.decompress(raw)
    else:
        raise PermanentError(f"unsupported compression type {info.compress_type}")
    if zlib.crc32(data) != info.CRC:
        raise RuntimeError("CRC-32 mismatch")
    return data


def download_audio(workers: int) -> None:
    wanted = load_wanted_ids()
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    log = open(AUDIO_LOG, "a")

    def log_line(msg: str) -> None:
        with lock:
            log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n")
            log.flush()

    lock = threading.Lock()
    todo = [tid for tid in wanted if not audio_path(tid).exists()]
    print(f"{len(wanted)} clips wanted, {len(wanted) - len(todo)} already present, {len(todo)} to fetch")
    if not todo:
        return

    print("Reading the fma_large.zip central directory over HTTP ...")
    with RemoteZip(LARGE_URL) as rz:
        members = {int(i.filename.rsplit("/", 1)[-1][:-4]): i for i in rz.infolist() if i.filename.endswith(".mp3")}
    print(f"  {len(members)} mp3 members in the archive")

    missing = [tid for tid in todo if tid not in members]
    for tid in missing:
        log_line(f"MISSING {tid} not in fma_large.zip")
    todo = [tid for tid in todo if tid in members]

    local = threading.local()

    def fetch(tid: int) -> int:
        if not hasattr(local, "session"):
            local.session = requests.Session()
        data = b""
        for attempt in range(4):
            try:
                data = _fetch_member(local.session, members[tid])
                break
            except (requests.RequestException, RuntimeError, zlib.error, OSError) as exc:
                if attempt == 3 or isinstance(exc, PermanentError):
                    raise
                log_line(f"RETRY {tid} attempt {attempt + 1}: {exc}")
                time.sleep(2**attempt)
        dst = audio_path(tid)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.rename(dst)
        return len(data)

    start, done, failed, nbytes = time.time(), 0, 0, 0
    last_print = 0.0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, tid): tid for tid in todo}
        for fut in as_completed(futures):
            tid = futures[fut]
            try:
                size = fut.result()
                nbytes += size
                if size == 0:
                    log_line(f"EMPTY {tid} zero-byte member")
            except Exception as exc:  # noqa: BLE001 - log every failure and keep going
                failed += 1
                log_line(f"FAILED {tid}: {exc}")
            done += 1
            now = time.time()
            if now - last_print > 10 or done == len(todo):
                last_print = now
                rate = done / (now - start)
                eta = (len(todo) - done) / rate if rate > 0 else float("inf")
                print(
                    f"  {done}/{len(todo)} clips, {failed} failed, {human_bytes(nbytes)} "
                    f"({human_bytes(nbytes / (now - start))}/s), ETA {eta / 60:.0f} min",
                    flush=True,
                )
    log.close()
    usage = sum(p.stat().st_size for p in AUDIO_DIR.rglob("*.mp3"))
    print(f"Done. {len(missing)} missing from archive, {failed} failed (see {AUDIO_LOG}).")
    print(f"Disk usage of {AUDIO_DIR}: {human_bytes(usage)}")


def load_wanted_ids() -> list[int]:
    return sorted(int(t) for t in pd.read_parquet(TARGETS_PATH).index)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m training.download", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("metadata", help="download and verify fma_metadata.zip, extract the CSVs")
    sub.add_parser("targets", help="build data/targets.parquet from the CSVs")
    p_audio = sub.add_parser("audio", help="fetch the needed clips from fma_large.zip")
    p_audio.add_argument("--workers", type=int, default=16, help="concurrent range requests")
    args = parser.parse_args()

    if args.cmd == "metadata":
        download_metadata()
    elif args.cmd == "targets":
        build_targets()
    else:
        download_audio(args.workers)


if __name__ == "__main__":
    main()
