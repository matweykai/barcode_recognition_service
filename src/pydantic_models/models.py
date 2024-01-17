from pydantic import BaseModel


class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class BarcodeDetection(BaseModel):
    bbox: BoundingBox
    text: str
