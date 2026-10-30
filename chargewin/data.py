"""Dataset download, checks and extraction."""
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

MANIFEST = json.loads((Path(__file__).with_name("data_manifest.json")).read_text(encoding="utf-8"))
AGENT = {"User-Agent": "charge-window-capacity/1.0 (+https://github.com/ArYan001500/charge-window-capacity)"}
QUICK_HUST_CELLS = ["1-1", "1-5", "2-2", "2-5", "3-1", "3-5", "4-1", "4-5", "5-1", "5-5", "6-1", "6-5", "7-1", "7-5"]


class MissingData(RuntimeError):
    pass


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _get(url, dest, size):
    part = dest.with_suffix(dest.suffix + ".partial")
    done = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={**AGENT, **({"Range": f"bytes={done}-"} if done else {})})
    with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if done else "wb") as f:
        last = time.monotonic()
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            done += len(b)
            if time.monotonic() - last > 10:
                print(f"  {dest.name}: {done / 1e6:.0f} of {size / 1e6:.0f} MB", flush=True)
                last = time.monotonic()
    part.replace(dest)


def check(item, data_dir, verify=True):
    path = Path(data_dir) / item["path"]
    if not path.exists():
        return False
    if path.stat().st_size != item["size"]:
        raise MissingData(f"{path}: size {path.stat().st_size}, expected {item['size']}")
    if verify and md5(path) != item["md5"]:
        raise MissingData(f"{path}: MD5 differs from the published file")
    return True


def prepare(datasets, data_dir, offline=False):
    """Download what can be downloaded, check the rest, and unpack the KIT archives."""
    data_dir = Path(data_dir)
    missing = []
    for item in MANIFEST["files"]:
        if item["dataset"] not in datasets:
            continue
        path = data_dir / item["path"]
        if check(item, data_dir):
            print(f"ok   {item['path']}", flush=True)
            continue
        if item.get("url") and not offline:
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"get  {item['path']} ({item['size'] / 1e6:.0f} MB)", flush=True)
            _get(item["url"], path, item["size"])
            if not check(item, data_dir):
                raise MissingData(f"{path}: download failed")
            continue
        missing.append(item)
    if missing:
        lines = [f"  {m['path']}  ({m['size'] / 1e9:.2f} GB) from {m['page']}" for m in missing]
        raise MissingData("these files must be downloaded by hand (see README, Data):\n" + "\n".join(lines))
    if "kit" in datasets:
        unpack_kit(data_dir)


def unpack_kit(data_dir):
    kit = Path(data_dir) / "kit"
    res = kit / "result_data"
    if not (res / "cfg.zip").exists() or not (res / "cell_eocv2.zip").exists():
        res.mkdir(parents=True, exist_ok=True)
        with tarfile.open(kit / "10.35097-1969.tar") as t:
            for m in t.getmembers():
                if m.name.endswith(("/cfg.zip", "/cell_eocv2.zip")):
                    with t.extractfile(m) as src, open(res / Path(m.name).name, "wb") as dst:
                        shutil.copyfileobj(src, dst)
    logs = kit / "cell_log_age_30s"
    if len(list(logs.glob("cell_log_age_30s_P*.csv"))) < 240:
        logs.mkdir(parents=True, exist_ok=True)
        archive = kit / "cell_log_age_ultracompr.7z"
        exe = shutil.which("7z") or shutil.which("7za") or next(
            (p for p in [r"C:\Program Files\7-Zip\7z.exe"] if Path(p).exists()), None)
        print("unpacking the 30-s KIT logs (reads the whole 7 GB archive, about 30 min)", flush=True)
        if exe:
            subprocess.run([exe, "e", str(archive), f"-o{logs}", "cell_log_age_30s_P*.csv", "-y"], check=True,
                           stdout=subprocess.DEVNULL)
        else:
            import py7zr
            with py7zr.SevenZipFile(archive) as z:
                names = [n for n in z.getnames() if Path(n).name.startswith("cell_log_age_30s_P")]
                z.extract(path=logs, targets=names)
            for f in logs.rglob("cell_log_age_30s_P*.csv"):
                if f.parent != logs:
                    f.replace(logs / f.name)
    n = len(list(logs.glob("cell_log_age_30s_P*.csv")))
    if n != 240:
        raise MissingData(f"{logs}: {n} log files, expected 240")


class RangeFile(io.RawIOBase):
    """Read-only file object over HTTP range requests, so that single zip members can be fetched."""

    def __init__(self, url):
        self.source = url
        self._resolve()
        self.pos = 0

    def _resolve(self):
        req = urllib.request.Request(self.source, headers={**AGENT, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.url, self.size = r.geturl(), int(r.headers["Content-Range"].split("/")[1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.url, headers={**AGENT, "Range": f"bytes={self.pos}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
                self._resolve()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def quick_hust(data_dir, offline=False):
    """Fetch 14 HUST cells from the published zip and store them as a small zip with the same layout."""
    dest = Path(data_dir) / "hust" / "our_data.zip"
    item = next(m for m in MANIFEST["files"] if m["dataset"] == "hust")
    want = {f"our_data/{c}.pkl": MANIFEST["quick_hust_md5"][c] for c in QUICK_HUST_CELLS}
    if dest.exists():
        with zipfile.ZipFile(dest) as z:
            if sorted(n for n in z.namelist() if n.endswith(".pkl")) == sorted(want) and all(
                    hashlib.md5(z.read(n)).hexdigest() == h for n, h in want.items()):
                print(f"ok   {dest}", flush=True)
                return dest
    if offline:
        raise MissingData(f"{dest} is missing; run without --offline")
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = zipfile.ZipFile(io.BufferedReader(RangeFile(item["url"]), buffer_size=1 << 20))
    part = dest.with_suffix(".zip.partial")
    with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED) as out:
        for name, h in want.items():
            print(f"get  {name} ({src.getinfo(name).compress_size / 1e6:.0f} MB)", flush=True)
            data = src.read(name)
            if hashlib.md5(data).hexdigest() != h:
                raise MissingData(f"{name}: MD5 differs from the published file")
            out.writestr(name, data)
    part.replace(dest)
    return dest
