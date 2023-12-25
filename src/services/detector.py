import cv2
import numpy as np
from onnxruntime import InferenceSession


class BarcodeDetector:
    def __init__(
        self,
        onnx_weights_path: str,
    ):
        """Inference class for YOLO model converted to ONNX

        Args:
            onnx_weights_path (str): path to model weights in ONNX format
        """
        self.session = InferenceSession(
            onnx_weights_path,
            providers=['CPUExecutionProvider'],
        )

        inp_shape = self.session.get_inputs().shape
        self._img_size = (inp_shape[3], inp_shape[2])

    def preprocess(self, raw_img: np.ndarray) -> np.ndarray:
        """Preprocess raw BGR image and make it ready for the inference

        Args:
            raw_img (np.ndarray): raw BGR image

        Returns:
            np.ndarray: ndarray of shape [1, 3, H, W] ready for the inference
        """
        prep_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        raw_img_width = raw_img.shape[1]
        raw_img_height = raw_img.shape[0]

        if raw_img_height > self._img_size[0] and raw_img_width > self._img_size[1]:
            prep_img = cv2.resize(prep_img, self._img_size, interpolation=cv2.INTER_AREA)
        else:
            prep_img = cv2.resize(prep_img, self._img_size, interpolation=cv2.INTER_LINEAR)

        # Convert HxWxC to CxHxW
        prep_img = np.transpose(prep_img, (2, 0, 1))

        return np.expand_dims(prep_img, axis=0)

    def detect(self, inp_img: np.ndarray):
        pass
