"""Django settings for the Fashion Try-On web frontend.

Minimal: SQLite for sessions, one local app (`tryon`), and an API_BASE_URL
that points the `tryon` views at the FastAPI service.

Loads `<repo-root>/.env` so settings can read the same secrets that FastAPI
uses (currently only API_BASE_URL is read here; Gemini/Groq keys live on
the API side).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")

SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "dev-only-secret-key-change-me-before-prod",
)
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = ["*"] if DEBUG else os.environ.get("DJANGO_ALLOWED_HOSTS", "").split(",")

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "tryon",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

# 25 MB cap on uploaded photos before Django spools to disk; protects against
# accidental multi-GB uploads. Real photos are 1–5 MB.
DATA_UPLOAD_MAX_MEMORY_SIZE = 25 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 25 * 1024 * 1024

# Where to find the FastAPI service. Both processes run on localhost in dev.
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8001")

# Where the static garment catalog lives (shared with FastAPI). Read-only here.
CLOTH_DIR = PROJECT_ROOT / "cloth"
CATALOG_PATH = CLOTH_DIR / "garments.json"

# Where FastAPI writes generated try-on PNGs. Both processes share this dir
# so Django can serve the result directly without proxying through FastAPI.
TRYON_OUTPUT_DIR = PROJECT_ROOT / "output"
TRYON_OUTPUT_DIR.mkdir(exist_ok=True)

# Public URL prefixes used by the templates / view URL builders.
CLOTH_URL = "/cloth/"
TRYON_OUTPUT_URL = "/tryon-output/"
