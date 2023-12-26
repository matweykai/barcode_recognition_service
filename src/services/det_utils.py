import numpy as np


def rescale_boxes(
    predictions: np.ndarray,
    img_size: tuple,
) -> np.ndarray:
    """Rescales YOLO boxes to the original image size

    Args:
        predictions (np.ndarray): raw YOLO predictions
        img_size (tuple): size of the target image

    Returns:
        np.ndarray: rescaled boxes
    """
    width, height = img_size

    scale_vector = np.array([width, height, width, height])

    mod_pred = np.zeros_like(predictions)
    mod_pred[:, 0] = mod_pred[:, 0] - mod_pred[:, 2] / 2
    mod_pred[:, 1] = mod_pred[:, 1] - mod_pred[:, 3] / 2
    mod_pred[:, 2] = mod_pred[:, 0] + mod_pred[:, 2] / 2
    mod_pred[:, 3] = mod_pred[:, 1] + mod_pred[:, 3] / 2

    return mod_pred * scale_vector


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

    while order:
        cur_ind = order[-1]

        keep.append(predictions[cur_ind])

        order = order[:-1]

        if order:
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
