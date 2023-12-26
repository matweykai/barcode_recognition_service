import cv2
import numpy as np


def resize_pad(img: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Resize image and add black padding where it is necessery

    Args:
        img (np.ndarray): input cv2 image
        target_width (int): target width of the image
        target_height (int): target height of the image

    Returns:
        np.ndarray: resized image with black padding to the right
    """
    img_h, img_w = img.shape[:2]

    tmp_width = min(int(img_w * target_height / img_h), target_width)

    img = cv2.resize(img, (tmp_width, target_height))
    width_diff = target_width - tmp_width

    if width_diff > 0:
        img = cv2.copyMakeBorder(img, 0, 0, 0, width_diff, cv2.BORDER_CONSTANT, value=0)

    return img
