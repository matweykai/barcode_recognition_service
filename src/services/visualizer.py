import cv2
import numpy as np

from src.pydantic_models.models import BarcodeDetection


class BarcodesVisualizer:
    def __init__(
        self,
        box_color: tuple,
        box_thickness: int,
        text_color: tuple,
        text_font_scale: float,
        text_thickness: int,
        text_y_shift: int,
    ):
        """Visualizes boxes and labels on the image

        Args:
            box_color (tuple): color of the bounding box
            box_thickness (int): thickness of the bounding box
            text_color (tuple): color of the label text
            text_font_scale (float): font scale of the label text
            text_thickness (int): thickness of the label text
            text_y_shift (int): label text vertical shift
        """
        self.box_color = box_color
        self.box_thickness = box_thickness
        self.text_color = text_color
        self.text_font_scale = text_font_scale
        self.text_thickness = text_thickness
        self.text_y_shift = text_y_shift

    def draw_detections(
        self,
        img: np.ndarray,
        detections_list: list[BarcodeDetection],
    ) -> np.ndarray:
        """Draws detections on the image

        Args:
            img (np.ndarray): raw image in BGR format
            detections_list (list[BarcodeDetection]): list of barcode detections

        Returns:
            np.ndarray: result image with drawn boxes and labels
        """
        for barcode_obj in detections_list:
            bbox = barcode_obj.bbox
            cv2.rectangle(
                img=img,
                pt1=(bbox.x_min, bbox.y_min),
                pt2=(bbox.x_max, bbox.y_max),
                color=self.box_color,
                thickness=self.box_thickness,
            )

            cv2.putText(
                img=img,
                text=barcode_obj.text,
                org=(bbox.x_min, bbox.y_min + self.text_y_shift),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=self.text_font_scale,
                color=self.text_color,
                thickness=self.text_thickness,
                lineType=cv2.LINE_AA,
            )

        return img
