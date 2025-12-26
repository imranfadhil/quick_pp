import logging
import os

from quick_pp.app.backend.task_queue.celery_app import celery_app
from quick_pp.database import objects as db_objects
from quick_pp.database.db_connector import DBConnector


@celery_app.task(bind=True, name="quick_pp.tasks.process_las", acks_late=True)
def process_las(self, saved_paths, project_id, depth_uom="m"):
    """Process uploaded LAS files and add them into the project.

    This task runs in a worker process and creates its own DBConnector
    instance to avoid inheriting engines from the webserver process.
    """
    try:
        connector = DBConnector(db_url=os.environ.get("QPP_DATABASE_URL"))
        with connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            result = proj.read_las(saved_paths, depth_uom=depth_uom)

        # If read_las returns a summary dict, inspect processed_files
        if isinstance(result, dict):
            processed = int(result.get("processed_files", 0))
            if processed == 0:
                # No files processed — treat this as a failure so the task is marked failed
                raise RuntimeError("No LAS files were successfully parsed")
            return {"status": "success", "summary": result}

        # Fallback: if read_las didn't return structured info, assume success
        return {
            "status": "success",
            "files": [os.path.basename(p) for p in saved_paths],
        }
    except Exception as exc:
        logging.getLogger(__name__).exception("Celery LAS processing failed: %s", exc)
        # Re-raise so Celery marks the task as failed and stores the traceback
        raise
