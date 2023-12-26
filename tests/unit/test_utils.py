import pytest
import numpy as np

from src.services.det_utils import non_max_suppression


@pytest.mark.parametrize(
    ('thres'),
    [item for item in np.arange(0.1, 1, 0.1)],
)
def test_nms_non_overlapping(thres: float):
    inp_boxes = np.array([[0, 0, 1, 1, 0.8], [1, 1, 2, 2, 0.7], [0, 1, 1, 2, 0.9], [1, 0, 2, 1, 0.85]])
    expected_boxes = [[0, 1, 1, 2, 0.9], [1, 0, 2, 1, 0.85], [0, 0, 1, 1, 0.8], [1, 1, 2, 2, 0.7]]

    result = non_max_suppression(inp_boxes, thres)

    assert (np.array(result) == np.array(expected_boxes)).all()


@pytest.mark.parametrize(
    ('thres'), 
    [item for item in np.arange(0.1, 1, 0.1)],
)
def test_nms_fully_overlapping(thres: float):
    inp_boxes = np.array([[0, 0, 1, 1, 0.8], [0, 0, 1, 1, 0.7], [0, 0, 1, 1, 0.9], [0, 0, 1, 1, 0.85]])
    expected_boxes = [[0, 0, 1, 1, 0.9]]

    result = non_max_suppression(inp_boxes, thres)

    assert (np.array(result) == np.array(expected_boxes)).all()


@pytest.mark.parametrize(
    ('thres', 'expected_boxes'), 
    [
        (0.9, [[0, 0, 1, 1, 0.9], [0.5, 0, 1.5, 1, 0.85], [0.5, 0.5, 1.5, 1.5, 0.7]]),
        (0.5, [[0, 0, 1, 1, 0.9], [0.5, 0, 1.5, 1, 0.85], [0.5, 0.5, 1.5, 1.5, 0.7]]),
        (0.3, [[0, 0, 1, 1, 0.9], [0.5, 0.5, 1.5, 1.5, 0.7]]),
        (0.1, [[0, 0, 1, 1, 0.9]]),
    ],
)
def test_nms_partly_overlapping(thres: float, expected_boxes: list):
    inp_boxes = np.array([[0, 0, 1, 1, 0.9], [0.5, 0.5, 1.5, 1.5, 0.7], [0.5, 0, 1.5, 1, 0.85]])

    result = non_max_suppression(inp_boxes, thres)

    assert (np.array(result) == np.array(expected_boxes)).all()
