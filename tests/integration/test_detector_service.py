import numpy as np


def test_real_img_inference(barcode_detector, barcode_data):
    barcode_img, ideal_pred = barcode_data

    pred = barcode_detector.detect(barcode_img)

    assert pred.shape[0] == 1
    assert np.all(np.abs(ideal_pred - pred[0, :4]) < 50)
