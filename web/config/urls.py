"""Top-level URL routing.

Most paths are delegated to the `tryon` app. In DEBUG, this file also
mounts two read-only static directories so the browser can fetch garment
images and finished try-on PNGs without round-tripping through FastAPI:

    /cloth/<path>          → <repo-root>/cloth/<path>
    /tryon-output/<path>   → <repo-root>/output/<path>     (FastAPI writes here)

In production, hand both off to a real reverse proxy / object store.
"""

from django.conf import settings
from django.conf.urls.static import static
from django.urls import include, path

urlpatterns = [
    path("", include("tryon.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.CLOTH_URL, document_root=settings.CLOTH_DIR)
    urlpatterns += static(settings.TRYON_OUTPUT_URL, document_root=settings.TRYON_OUTPUT_DIR)
