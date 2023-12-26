import cv2
import numpy as np
from onnxruntime import InferenceSession

from src.services.det_utils import non_max_suppression, rescale_boxes, convert_xywh_to_xyxy


class BarcodeDetector:
    def __init__(
        self,
        onnx_weights_path: str,
        iou_threshold: float = 0.6,
        conf_threshold: float = 0.6,
    ):
        """Inference class for YOLO model converted to ONNX

        Args:
            onnx_weights_path (str): path to model weights in ONNX format
        """
        self.session = InferenceSession(
            onnx_weights_path,
            providers=['CPUExecutionProvider'],
        )

        inp_shape = self.session.get_inputs()[0].shape
        self._img_size = (inp_shape[3], inp_shape[2])
        self._iou_threshold = iou_threshold
        self._conf_threshold = conf_threshold

    def detect(
        self,
        inp_img: np.ndarray,
    ) -> np.ndarray:
        """Detect barcode in the image

        Args:
            inp_img (np.ndarray): raw BGR image

        Returns:
            np.ndarray: coordinates of the detected barcodes
        """
        prep_img = self._preprocess(inp_img)

        model_pred = self.session.run(None, {'images': prep_img})[0]

        return self._postprocess(model_pred, (inp_img.shape[1], inp_img.shape[0]))

    def _preprocess(self, raw_img: np.ndarray) -> np.ndarray:
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

        # Normalization
        prep_img = (prep_img / 255.).astype(np.float32)

        return np.expand_dims(prep_img, axis=0)

    def _postprocess(
        self,
        model_output: np.ndarray,
        inp_img_size: tuple,
    ) -> np.ndarray:
        """Postprocess YOLO output, applies NMS and rescales bounding boxes

        Args:
            model_output (np.ndarray): raw model detections

        Returns:
            np.ndarray: postprocessed YOLO predictions
        """
        model_output = model_output[0]
        model_output = np.transpose(model_output, (1, 0))

        # Filter out boxes with small confidence
        res_pred = model_output[model_output[:, 4] > self._conf_threshold, :]

        res_pred = convert_xywh_to_xyxy(res_pred)
        res_pred = np.array(non_max_suppression(res_pred, self._iou_threshold))

        if res_pred.shape[0] > 0:
            res_pred = rescale_boxes(res_pred, self._img_size, inp_img_size)

        return res_pred
    

if __name__ == '__main__':
    detector = BarcodeDetector('/home/matwey_i/temp/pr_2/service/weights/yolo_8n_994.onnx', 0.6, 0.6)

    test_img = cv2.imread('/home/matwey_i/temp/pr_2/service/tests/images/test_img.jpg')

    pred = detector.detect(test_img)

    # test_img = cv2.resize(test_img, (640, 640))

    print(pred)

    for item in pred:
        cv2.rectangle(test_img, (int(item[0]), int(item[1])), (int(item[2]), int(item[3])), (0, 0, 255), 2)
        print(item)

    cv2.imwrite('temp_pred.jpg', test_img)
    
