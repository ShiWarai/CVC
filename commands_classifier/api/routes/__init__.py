"""Routes для API сервера CVC."""

from commands_classifier.api.routes.predict import router as predict_router
from commands_classifier.api.routes.training import router as training_router
from commands_classifier.api.routes.examples import router as examples_router
from commands_classifier.api.routes.package import router as package_router
from commands_classifier.api.routes.health import router as health_router

__all__ = [
    "predict_router",
    "training_router", 
    "examples_router",
    "package_router",
    "health_router",
]
