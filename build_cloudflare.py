"""
Build script to export PREDICTOR (VisionClassify) into `dist/` for Cloudflare Pages (`*.pages.dev`).

Exports:
- `/` -> `dist/index.html`
- `/classifier/` -> `dist/classifier/index.html`
- `/history/` -> `dist/history/index.html`
- `/about/` -> `dist/about/index.html`
- `/result/<id>/` -> `dist/result/<id>/index.html`
- `/manifest.json` -> `dist/manifest.json`
- `/service-worker.js` -> `dist/service-worker.js`
- `/api/sample-image/<category>/` -> `dist/api/sample-image/<category>/index.jpg` & `_redirects`
- `classifier/static/` -> `dist/static/`
- `media/` -> `dist/media/`
"""

import os
import shutil
from pathlib import Path

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django
django.setup()

from django.test import Client
from classifier.models import ClassificationHistory

BASE_DIR = Path(__file__).resolve().parent
DIST_DIR = BASE_DIR / "dist"


def export_cloudflare_bundle() -> None:
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    client = Client()

    routes = {
        "/": DIST_DIR / "index.html",
        "/classifier/": DIST_DIR / "classifier" / "index.html",
        "/history/": DIST_DIR / "history" / "index.html",
        "/about/": DIST_DIR / "about" / "index.html",
        "/manifest.json": DIST_DIR / "manifest.json",
        "/service-worker.js": DIST_DIR / "service-worker.js",
    }

    for rec in ClassificationHistory.objects.all():
        routes[f"/result/{rec.pk}/"] = DIST_DIR / "result" / str(rec.pk) / "index.html"

    for route, dest in routes.items():
        resp = client.get(route)
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = b"".join(resp.streaming_content) if getattr(resp, "streaming", False) else resp.content
        dest.write_bytes(content)
        print(f"[+] Exported {route} -> {dest.relative_to(BASE_DIR)}")

    # Export sample images
    for cat in ("cat", "dog", "automobile", "airplane", "flower"):
        resp = client.get(f"/api/sample-image/{cat}/")
        sample_dest = DIST_DIR / "api" / "sample-image" / cat / "index.html"
        sample_dest.parent.mkdir(parents=True, exist_ok=True)
        sample_dest.write_bytes(resp.content)

    # Copy static assets
    static_src = BASE_DIR / "classifier" / "static" / "classifier"
    static_dst = DIST_DIR / "static" / "classifier"
    shutil.copytree(static_src, static_dst, dirs_exist_ok=True)

    # Copy media uploads
    media_src = BASE_DIR / "media"
    media_dst = DIST_DIR / "media"
    if media_src.exists():
        shutil.copytree(media_src, media_dst, dirs_exist_ok=True)

    # Create Cloudflare Pages _headers for correct MIME types on sample images
    headers_content = """/api/sample-image/*
  Content-Type: image/jpeg
/service-worker.js
  Service-Worker-Allowed: /
  Cache-Control: no-cache
"""
    (DIST_DIR / "_headers").write_text(headers_content, encoding="utf-8")
    print(f"[+] Successfully built Cloudflare Pages bundle in {DIST_DIR}")


if __name__ == "__main__":
    export_cloudflare_bundle()
