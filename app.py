import uvicorn
from fastapi import FastAPI

from src.containers.container import AppContainer
from src.routes.routers import router as app_router
from src.pydantic_models.settings import Settings
from src.routes import barcode_routes


def create_app(settings: Settings) -> FastAPI:
    container = AppContainer()
    container.config.from_dict(settings.model_dump())
    container.wire([barcode_routes])

    app = FastAPI()

    set_routers(app)

    return app


def set_routers(app: FastAPI):
    app.include_router(app_router, prefix='/barcodes')


if __name__ == '__main__':
    settings = Settings.from_yaml('config/app_config.yaml')

    app = create_app(settings)

    app_settings = settings.app_settings
    uvicorn.run(app, port=app_settings.port, host=app_settings.host)
