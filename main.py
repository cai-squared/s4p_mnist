"""Cloud Functions Gen 2 entry point (gcloud --entry-point=handler)."""

from mangum import Mangum

from api.main import app

handler = Mangum(app, lifespan="off")

__all__ = ["handler"]
