import pytest
import numpy as np

from src.services.detector import BarcodeDetector


@pytest.mark.parametrize(
    ('test_size', 'expected_size'),
    [
        ((400, 500), (100, 200)),
        ((50, 50), (100, 200)),
        ((10, 20), (100, 200)),
        ((100, 200), (100, 200)),
    ]
)
def test_preprocessing(test_size, expected_size, barcode_detector, mocker):
    test_img = np.random.randint(0, 256, size=(test_size[0], test_size[1], 3)).astype(np.uint8)

    prep_img = barcode_detector.preprocess(test_img)

    assert prep_img.shape == (1, 3, expected_size[0], expected_size[1])
