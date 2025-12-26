import os

from celery import Celery

# Configure broker and backend via environment variables. Defaults assume a
# local Redis instance — adjust in production (RabbitMQ, etc.).
broker = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

celery_app = Celery("quick_pp", broker=broker, backend=backend)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

# Import tasks so they are registered with the Celery app when a worker starts.
try:
    # Local import to avoid circular import at module import time for other paths
    from . import tasks  # noqa: F401
except Exception:
    try:
        # Fallback to absolute import
        import quick_pp.app.backend.task_queue.tasks  # noqa: F401
    except Exception:
        pass


def is_broker_available(timeout: float = 1.0) -> bool:
    """Quick check whether the Celery broker is reachable.

    Uses the underlying kombu connection context to attempt a connection.
    Returns True when a connection can be established, False otherwise.
    """
    try:
        with celery_app.connection_for_read() as conn:
            # Accessing default_channel triggers a real connection
            _ = conn.default_channel
        return True
    except Exception:
        return False
