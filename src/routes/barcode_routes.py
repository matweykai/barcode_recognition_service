import cv2
import numpy as np
from fastapi import File, Depends, Response
from dependency_injector.wiring import Provide, inject

from src.routes.routers import router
from src.schemas.barcode_routes import FindBarcodeAnswer

from src.services.analyzer import BarcodeAnalyzer
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
