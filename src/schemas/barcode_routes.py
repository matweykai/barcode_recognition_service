from pydantic import BaseModel

from src.pydantic_models.models import BarcodeDetection


class FindBarcodeAnswer(BaseModel):
    barcodes: list[BarcodeDetection]
    image: bytes | None = None
