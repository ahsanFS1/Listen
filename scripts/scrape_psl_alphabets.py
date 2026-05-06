"""Scrape sign demonstration videos for the 40 letters in the Urdu
alphabet category from psl.org.pk.

Strategy mirrors scrape_psl_videos.py but for category 8 (Urdu Alphabet):
  1. GET https://admin.psl.org.pk/api/category/8 → list of 40 concepts.
  2. For each concept, GET /api/concept/{id}/videos → first video URL.
  3. Download the mp4 (CloudFront-hosted) to
     flutter_app/assets/videos/letters/{urdu_glyph}.mp4 — Urdu glyph is
     the natural primary key (all 40 are unique) and lets the Flutter
     side derive the asset path directly from PslLetter.urdu without a
     manifest lookup.

Also writes flutter_app/assets/videos/letters/manifest.json for debugging:
    { "ا": { "id": 363, "slug": "alif", "title": "Alif",
             "asset": "assets/videos/letters/ا.mp4" }, ... }
"""
import json
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
OUT_DIR = ROOT / "flutter_app" / "assets" / "videos" / "letters"
CAT_API = "https://admin.psl.org.pk/api/category/8"
VID_TPL = "https://admin.psl.org.pk/api/concept/{cid}/videos"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"


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

    print(f"[INFO] fetching category list from {CAT_API}")
    cat = fetch_json(CAT_API)
    concepts = cat["data"]["concepts"]
    print(f"[INFO] {len(concepts)} concepts in Urdu Alphabet category")

    manifest_path = OUT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    failures = []

    for i, c in enumerate(concepts, 1):
        cid = c["id"]
        glyph = c["title_secondary"]
        slug = c["slug"]
        title = c["title"]
        out_mp4 = OUT_DIR / f"{glyph}.mp4"

        if out_mp4.exists() and out_mp4.stat().st_size > 10_000:
            manifest[glyph] = {
                "id": cid, "slug": slug, "title": title,
                "asset": f"assets/videos/letters/{glyph}.mp4",
            }
            print(f"[{i:02d}/{len(concepts)}] {glyph}  {title:<18s} cached ({out_mp4.stat().st_size//1024} KB)")
            continue

        try:
            data = fetch_json(VID_TPL.format(cid=cid))
            videos = data.get("data", {}).get("videos", [])
            if not videos:
                raise RuntimeError("api returned no videos")
            mp4_url = videos[0].get("video_url")
            if not mp4_url:
                raise RuntimeError("video_url missing")
            print(f"[{i:02d}/{len(concepts)}] {glyph}  {title:<18s} → {mp4_url}")
            download(mp4_url, out_mp4)
            manifest[glyph] = {
                "id": cid, "slug": slug, "title": title,
                "asset": f"assets/videos/letters/{glyph}.mp4",
            }
        except (HTTPError, URLError, RuntimeError) as e:
            print(f"[{i:02d}/{len(concepts)}] {glyph}  {title:<18s} FAIL ({e})")
            failures.append((title, glyph, str(e)))
        time.sleep(0.4)

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False)
    )
    print(f"\n[OK] manifest saved → {manifest_path}")
    print(f"[OK] {len(manifest)} videos available, {len(failures)} failures")
    for t, g, why in failures:
        print(f"     - {t} ({g}): {why}")


if __name__ == "__main__":
    sys.exit(main())
