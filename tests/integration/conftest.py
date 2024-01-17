import os
import cv2
import pytest
import numpy as np
from omegaconf import OmegaConf

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.detector import BarcodeDetector
from src.services.recognizer import BarcodeRecognizer
from src.services.analyzer import BarcodeAnalyzer
from src.containers.container import AppContainer
from src.pydantic_models.settings import Settings
from src.routes import barcode_routes
from app import set_routers


@pytest.fixture()
def barcode_data():
    return (cv2.imread('tests/images/test_img.jpg'), np.array([275,  842,  718, 1150]))


@pytest.fixture()
def crop_barcode_data():
    return (cv2.imread('tests/images/test_barcode.jpg'), '4607072715168')


@pytest.fixture(scope='session')
def barcode_detector():
    return BarcodeDetector(
        onnx_weights_path='weights/yolo_8n_994.onnx',
        iou_threshold=0.6,
        conf_threshold=0.6,
    )


@pytest.fixture(scope='session')
def barcode_recognizer():
    return BarcodeRecognizer(
        onnx_weights_path='weights/eff_net_b5_loss=0.196.onnx',
    )


@pytest.fixture(scope='session')
def barcode_analyzer(barcode_detector, barcode_recognizer):
    return BarcodeAnalyzer(
        detector=barcode_detector,
        recognizer=barcode_recognizer,
    )


@pytest.fixture(scope='session')
def single_barcode_img():
    with open(os.path.join('tests', 'images', 'test_img.jpg'), 'rb') as file:
        return file.read()
    

@pytest.fixture(scope='session')
def multiple_barcode_img():
    with open(os.path.join('tests', 'images', 'test_multiple_barcodes.jpg'), 'rb') as file:
        return file.read()


@pytest.fixture(scope='session')
def app_config():
    return Settings.from_yaml(os.path.join('tests', 'config', 'test_config.yaml'))


@pytest.fixture
def wired_app_container(app_config):
    container = AppContainer()
    container.config.from_dict(app_config.model_dump())
    container.wire([barcode_routes])
    
    yield container
    
    container.unwire()


@pytest.fixture
def test_app(wired_app_container):
    app = FastAPI()
    set_routers(app)
    
    return app


@pytest.fixture
def client(test_app):
    return TestClient(test_app)
