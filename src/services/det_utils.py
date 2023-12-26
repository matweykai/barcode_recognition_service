import numpy as np


def rescale_boxes(
    predictions: np.ndarray,
    old_size: tuple,
    new_size: tuple,
) -> np.ndarray:
    """Rescales YOLO boxes to the original image size

    Args:
        predictions (np.ndarray): raw YOLO predictions
        img_size (tuple): size of the target image

    Returns:
        np.ndarray: rescaled boxes
    """
    width_ratio = new_size[0] / old_size[0]
    height_ratio = new_size[1] / old_size[1]

    mod_pred = predictions.copy()
    mod_pred[:, [0, 2]] *= width_ratio
    mod_pred[:, [1, 3]] *= height_ratio

    return mod_pred


def convert_xywh_to_xyxy(inp_pred: np.ndarray) -> np.ndarray:
    """Converts xywh predictions to xyxy format

    Args:
        inp_pred (np.ndarray): input detections in xywh format

    Returns:
        np.ndarray: converted boxes to format xyxy
    """
    conv_pred = np.empty_like(inp_pred)

    dw = inp_pred[:, 2] / 2  # half-width
    dh = inp_pred[:, 3] / 2  # half-height
    conv_pred[..., 0] = inp_pred[..., 0] - dw  # top left x
    conv_pred[..., 1] = inp_pred[..., 1] - dh  # top left y
    conv_pred[..., 2] = inp_pred[..., 0] + dw  # bottom right x
    conv_pred[..., 3] = inp_pred[..., 1] + dh  # bottom right y
    conv_pred[..., 4] = inp_pred[..., 4]

    return conv_pred


def non_max_suppression(
    predictions: np.ndarray,
    iou_threshold: float = 0.5,
) -> list:
    """NMS for output bounding boxes

    Args:
        predictions (np.ndarray): array of bounding boxes in format [x_min, y_min, x_max, y_max]
        iou_threshold (float, optional): threshold for IOU. Defaults to 0.5.

    Returns:
        list: usefull bounding boxes
    """
    x_min = predictions[:, 0]
    y_min = predictions[:, 1]
    x_max = predictions[:, 2]
    y_max = predictions[:, 3]

    scores = predictions[:, 4]

    areas = (y_max - y_min) * (x_max - x_min)

    order = scores.argsort()

    keep = []

    while order.shape[0] > 0:
        cur_ind = order[-1]

        keep.append(predictions[cur_ind])

        order = order[:-1]

        if order.shape[0] == 0:
            break

        # Find boxes intersection
        x_min_inter = np.maximum(x_min[order], x_min[cur_ind])
        y_min_inter = np.maximum(y_min[order], y_min[cur_ind])
        x_max_inter = np.minimum(x_max[order], x_max[cur_ind])
        y_max_inter = np.minimum(y_max[order], y_max[cur_ind])

        # Calculate Intersection Over Union
        inter_width = np.clip(x_max_inter - x_min_inter, 0, None)
        inter_height = np.clip(y_max_inter - y_min_inter, 0, None)

        inter_area = inter_height * inter_width
        union = areas[order] - inter_area + areas[cur_ind]
        iou_arr = inter_area / union

        # Drop highly overlapped boxes
        mask = iou_arr < iou_threshold
        order = order[mask]

    return keep
