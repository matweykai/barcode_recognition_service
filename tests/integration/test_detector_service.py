import cv2
import numpy as np

from src.services.detector import BarcodeDetector


def test_real_img_inference():
    test_img = cv2.imread('tests/images/test_img.jpg')
    # ideal_pred = np.array([497, 996, 443, 308])
    ideal_pred = np.array([275,  842,  718, 1150])


    detector = BarcodeDetector(
        onnx_weights_path='weights/yolo_8n_994.onnx',
        iou_threshold=0.6,
        conf_threshold=0.6,
    )

    pred = detector.detect(test_img)

    assert pred.shape[0] == 1
    assert np.all(np.abs(ideal_pred - pred[0, :4]) < 50)
