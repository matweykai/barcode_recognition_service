import cv2
import numpy as np
from onnxruntime import InferenceSession

from src.services.rec_utils import resize_pad, matrix_to_string


class BarcodeRecognizer:
    def __init__(
        self,
        onnx_weights_path: str,
    ):
        self.session = InferenceSession(
            onnx_weights_path,
            providers=['CPUExecutionProvider'],
        )

        inp_shape = self.session.get_inputs()[0].shape
        self._img_size = (inp_shape[3], inp_shape[2])
        self._vocab = '0123456789'

    def decode(self, inp_img: np.ndarray) -> np.ndarray:
        pass

    def _preprocess(self, raw_img: np.ndarray) -> np.ndarray:
        """Preprocess raw BGR image and make it ready for the inference

        Args:
            raw_img (np.ndarray): raw BGR image

        Returns:
            np.ndarray: ndarray of shape [1, 3, H, W] ready for the inference
        """
        prep_img = resize_pad(raw_img, self._img_size[0], self._img_size[1])
        prep_img = cv2.cvtColor(prep_img, cv2.COLOR_BGR2RGB)

        # Normalize img
        max_pixel_val = 255.0
        prep_img = (prep_img / max_pixel_val).astype(np.float32)
        prep_img -= np.array((0.485, 0.456, 0.406))
        prep_img /= np.array((0.229, 0.224, 0.225))

        prep_img = np.transpose(prep_img, (2, 0, 1))

        return np.expand_dims(prep_img, axis=0)

    def _postprocess(self, model_pred: np.ndarray) -> str:
        """Postprocessing of the CRNN model output

        Args:
            model_pred (np.ndarray): raw CRNN predictions

        Returns:
            str: decoded barcode
        """
        return matrix_to_string(model_pred, self._vocab)[0][0]
