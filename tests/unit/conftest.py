import pytest

from src.services.detector import BarcodeDetector


@pytest.fixture()
def barcode_detector(mocker):
    shape_mock = mocker.PropertyMock()
    shape_mock.shape = (1, 3, 100, 200)

    mocker.patch('src.services.detector.InferenceSession.__init__', return_value=None)
    mocker.patch('src.services.detector.InferenceSession.get_inputs', return_value=[shape_mock])

    return BarcodeDetector('some_path.onnx')
