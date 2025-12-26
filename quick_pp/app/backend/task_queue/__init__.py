"""Celery task queue package for quick_pp."""

from .celery_app import celery_app  # re-export for convenience

__all__ = ["celery_app"]
