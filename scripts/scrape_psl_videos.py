"""Scrape sign demonstration videos from psl.org.pk for the 64 PSL words
the Listen model recognises.

Strategy: psl.org.pk is a Next.js client-rendered SPA that fetches video
metadata from `https://admin.psl.org.pk/api/concept/{concept_id}/videos`.
The HTML pages are not enough; we must call the JSON API. Each curated
URL in `flutter_app/lib/data/signs.dart` ends with `/{concept_id}-{slug}`,
so we extract concept_id from the URL, hit the API, then download the
returned mp4 (CloudFront-hosted) into `flutter_app/assets/videos/`.

After download, writes `flutter_app/assets/videos/manifest.json`:
    { "<word_id>": "assets/videos/<word_id>.mp4", ... }
"""
import json
import re
import ssl
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()

ROOT = Path(__file__).resolve().parents[1]
SIGNS_DART = ROOT / "flutter_app" / "lib" / "data" / "signs.dart"
OUT_DIR = ROOT / "flutter_app" / "assets" / "videos"
API_TPL = "https://admin.psl.org.pk/api/concept/{cid}/videos"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"


def parse_signs_urls(dart_path: Path) -> dict[str, str]:
    """Pull `'word_id': 'https://psl.org.pk/...'` pairs out of signs.dart."""
    txt = dart_path.read_text()
    block = txt.split("const Map<String, String> kSignUrls", 1)[1]
    block = block.split("};", 1)[0]
    pat = re.compile(r"'([^']+)'\s*:\s*'(https://psl\.org\.pk/[^']+)'")
    return dict(pat.findall(block))


def concept_id_from_url(url: str) -> int | None:
    m = re.search(r"/(\d+)-[a-z0-9!\-]+/?$", url)
    return int(m.group(1)) if m else None


def fetch_json(url: str) -> dict:
    req = Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urlopen(req, timeout=30, context=SSL_CTX) as r:
        return json.loads(r.read())


def download(url: str, dst: Path):
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=120, context=SSL_CTX) as r, open(dst, "wb") as f:
        while True:
            chunk = r.read(64 * 1024)
            if not chunk:
                break
            f.write(chunk)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    urls = parse_signs_urls(SIGNS_DART)
    print(f"[INFO] {len(urls)} curated PSL Dict URLs found")

    manifest_path = OUT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    failures = []
    for i, (word_id, url) in enumerate(sorted(urls.items()), 1):
        out_mp4 = OUT_DIR / f"{word_id}.mp4"
        if out_mp4.exists() and out_mp4.stat().st_size > 10_000:
            manifest[word_id] = f"assets/videos/{word_id}.mp4"
            print(f"[{i:02d}/{len(urls)}] {word_id:22s} cached ({out_mp4.stat().st_size//1024} KB)")
            continue

        cid = concept_id_from_url(url)
        if cid is None:
            print(f"[{i:02d}/{len(urls)}] {word_id:22s} SKIP (no concept id in {url!r})")
            failures.append((word_id, "no_concept_id"))
            continue

        try:
            data = fetch_json(API_TPL.format(cid=cid))
            videos = data.get("data", {}).get("videos", [])
            if not videos:
                raise RuntimeError("api returned no videos")
            mp4_url = videos[0].get("video_url")
            if not mp4_url:
                raise RuntimeError("video_url missing")
            print(f"[{i:02d}/{len(urls)}] {word_id:22s} → {mp4_url}")
            download(mp4_url, out_mp4)
            manifest[word_id] = f"assets/videos/{word_id}.mp4"
        except (HTTPError, URLError, RuntimeError) as e:
            print(f"[{i:02d}/{len(urls)}] {word_id:22s} FAIL ({e})")
            failures.append((word_id, str(e)))
        # be polite to the API
        time.sleep(0.4)

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\n[OK] manifest saved → {manifest_path}")
    print(f"[OK] {len(manifest)} videos available, {len(failures)} failures")
    for w, why in failures:
        print(f"     - {w}: {why}")


if __name__ == "__main__":
    sys.exit(main())
