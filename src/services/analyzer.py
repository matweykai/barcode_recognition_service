import numpy as np

from src.pydantic_models.models import BarcodeDetection, BoundingBox
from src.services.detector import BarcodeDetector
from src.services.recognizer import BarcodeRecognizer


class BarcodeAnalyzer:
    def __init__(
        self,
        detector: BarcodeDetector,
        recognizer: BarcodeRecognizer,
        extra_crop_val: int = 10,
    ) -> None:
        """Analyzes image based on detector and recognizer models

        Args:
            detector (BarcodeDetector): BarcodeDetector instance
            recognizer (BarcodeRecognizer): BarcodeRecognizer instance
        """
        self.detector = detector
        self.recognizer = recognizer
        self.extra_crop_val = extra_crop_val

    def find_barcodes(self, raw_img: np.ndarray) -> list[BarcodeDetection]:
        """Finds barcodes in the image

        Args:
            raw_img (np.ndarray): raw BGR image

        Returns:
            list[BarcodeDetection]: list of detected Barcodes
        """
        barcodes_list = []

        barcode_detections = self.detector.detect(raw_img)

        for x_min, y_min, x_max, y_max in barcode_detections:
            x_min = max(0, int(x_min) - self.extra_crop_val)
            y_min = max(0, int(y_min) - self.extra_crop_val)
            x_max = min(raw_img.shape[1], int(x_max) + self.extra_crop_val)
            y_max = min(raw_img.shape[0], int(y_max) + self.extra_crop_val)

            img_crop = raw_img[y_min: y_max, x_min: x_max, :]

            decoded_text = self.recognizer.decode(img_crop)

            bbox = BoundingBox(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
            )

            barcodes_list.append(BarcodeDetection(
                bbox=bbox,
                text=decoded_text,
            ))

        return barcodes_list
