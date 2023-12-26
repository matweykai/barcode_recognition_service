import cv2
import pytest
import numpy as np

from src.services.detector import BarcodeDetector
from src.services.recognizer import BarcodeRecognizer
from src.services.analyzer import BarcodeAnalyzer


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
