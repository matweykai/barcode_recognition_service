import numpy as np


def test_find_barcodes(barcode_analyzer):
    test_img = np.random.randint(0, 256, size=(1024, 1024, 3)).astype(np.uint8)

    barcode_analyzer.find_barcodes(test_img)

    barcode_analyzer.detector.detect.assert_called_once()
    barcode_analyzer.recognizer.decode.assert_not_called()
