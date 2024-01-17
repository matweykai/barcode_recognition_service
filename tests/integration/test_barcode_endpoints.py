import cv2
import numpy as np
from fastapi.testclient import TestClient
from http import HTTPStatus


def test_find_barcodes_response(client: TestClient, single_barcode_img: bytes):
    files = {
        'image': single_barcode_img,
    }
    response = client.post('/barcodes/find_barcode', files=files)

    assert response.status_code == HTTPStatus.OK

    response_json = response.json()
    predicted_list = response_json['barcodes']

    assert isinstance(predicted_list, list)
    assert len(predicted_list) > 0


def test_find_barcodes_bbox_count(client: TestClient, single_barcode_img: bytes):
    files = {
        'image': single_barcode_img,
    }
    response = client.post('/barcodes/find_barcode', files=files)

    assert response.status_code == HTTPStatus.OK

    response_json = response.json()
    predicted_list = response_json['barcodes']

    assert len(predicted_list) == 1


def test_find_barcodes_multiple_bbox(client: TestClient, multiple_barcode_img: bytes):
    files = {
        'image': multiple_barcode_img,
    }
    response = client.post('/barcodes/find_barcode', files=files)

    assert response.status_code == HTTPStatus.OK

    response_json = response.json()
    predicted_list = response_json['barcodes']

    assert len(predicted_list) > 1


def test_visualize_barcodes_response(client: TestClient, single_barcode_img: bytes):
    files = {
        'image': single_barcode_img,
    }
    response = client.post('/barcodes/visualize_barcode', files=files)

    assert response.status_code == HTTPStatus.OK

    response_bytes = response.content

    # Decode bytes to images
    req_img = cv2.imdecode(np.frombuffer(single_barcode_img, np.uint8), cv2.IMREAD_COLOR)
    resp_img = cv2.imdecode(np.frombuffer(response_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Check that rectangle was drawn
    img_diff = cv2.subtract(req_img, resp_img)
    img_diff = cv2.cvtColor(img_diff, cv2.COLOR_BGR2GRAY)
    _, img_diff = cv2.threshold(img_diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cnts, _ = cv2.findContours(img_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Get borders of the rectangle
    min_rect = cv2.minAreaRect(cnts[0])

    # Check that rectangle is big
    assert cv2.contourArea(cnts[0]) > 30 ** 2
    # Check that rectangle is horizontal
    assert abs(min_rect[2]) < 2
