import pytest
import numpy as np


@pytest.mark.parametrize(
    ('test_size', 'expected_size'),
    [
        ((400, 500), (96, 416)),
        ((50, 50), (96, 416)),
        ((10, 20), (96, 416)),
        ((100, 200), (96, 416)),
    ]
)
def test_preprocessing(test_size, expected_size, barcode_recognizer):
    test_img = np.random.randint(0, 256, size=(test_size[0], test_size[1], 3)).astype(np.uint8)

    prep_img = barcode_recognizer._preprocess(test_img)

    assert prep_img.shape == (1, 3, expected_size[0], expected_size[1])
