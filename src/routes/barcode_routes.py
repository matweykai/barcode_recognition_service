import cv2
import numpy as np
from fastapi import File, Depends, Response
from dependency_injector.wiring import Provide, inject

from src.routes.routers import router
from src.schemas.barcode_routes import FindBarcodeAnswer

from src.services.analyzer import BarcodeAnalyzer
from src.services.visualizer import BarcodesVisualizer
from src.containers.container import AppContainer


@router.post('/find_barcode')
@inject
def find_barcodes(
    image: bytes = File(),
    service: BarcodeAnalyzer = Depends(Provide[AppContainer.barcode_analyzer]),
) -> FindBarcodeAnswer:
    """Endpoint for using BarcodeAnalyzer on the image

    Args:
        image (bytes, optional): input image

    Returns:
        FindBarcodeAnswer: result of image analyze as json
    """
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)

    return FindBarcodeAnswer(barcodes=service.find_barcodes(img))


@router.post('/visualize_barcode', response_class=Response)
@inject
def visualize_barcodes(
    image: bytes = File(),
    service: BarcodeAnalyzer = Depends(Provide[AppContainer.barcode_analyzer]),
    visualizer: BarcodesVisualizer = Depends(Provide[AppContainer.barcode_visualizer]),
) -> Response:
    """Endpoint for using BarcodeAnalyzer on the image and drawing results

    Args:
        image (bytes, optional): input image

    Returns:
        Response: image with drawn bboxes and labels
    """
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)

    barcodes_list = service.find_barcodes(img)
    img_with_boxes = visualizer.draw_detections(img, barcodes_list)

    img_bytes = cv2.imencode('.jpg', img_with_boxes)[1].tobytes()

    return Response(content=img_bytes, media_type='image/jpg')
