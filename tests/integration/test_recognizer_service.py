import cv2
import numpy as np

from src.services.recognizer import BarcodeRecognizer


def test_real_img_inference():
    test_img = cv2.imread('tests/images/test_barcode.jpg')

    recognizer = BarcodeRecognizer(
        onnx_weights_path='weights/eff_net_b5_loss=0.196.onnx',
    )

    pred = recognizer.decode(test_img)

    assert sum([item[0] == item[1] for item in zip(pred, '4607072715168')]) > 10
