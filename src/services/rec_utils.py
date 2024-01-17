import cv2
import operator
import itertools
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


def softmax(x: np.ndarray) -> np.ndarray:
    """Computes 1d sofrmax in x array

    Args:
        x (np.ndarray): input 1 dimensional array of logits

    Returns:
        np.ndarray: 1 dimensional array of probabilities
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def matrix_to_string(
    model_output: np.ndarray,
    vocab: str,
) -> tuple[list[str], list[np.ndarray]]:
    """Decodes CTC matrix to string

    Args:
        model_output (np.ndarray): CRNN model raw output
        vocab (str): vocabulary of available symbols

    Returns:
        tuple[list[str], list[np.ndarray]]: tuple that consists of list of decoded barcodes,
            list of probabilites of this symbols
    """
    labels, confs = postprocess(model_output)
    labels_decoded, conf_decoded = decode(labels_raw=labels, conf_raw=confs)
    string_pred = labels_to_strings(labels_decoded, vocab)

    return string_pred, conf_decoded


def postprocess(model_output: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Converts raw model output to confidences and labels

    Args:
        model_output (np.ndarray): raw CRNN model output

    Returns:
        tuple[np.ndarray, np.ndarray]: array of labels, array of confidences
    """
    output = model_output.transpose(1, 0, 2)
    output = np.apply_along_axis(softmax, 2, output)

    confidences = np.max(output, axis=2)
    labels = np.argmax(output, axis=2)

    return labels, confidences


def decode(
    labels_raw: np.ndarray,
    conf_raw: np.ndarray,
) -> tuple[list[list], list[np.ndarray]]:
    """Decodes array of labels and confidences

    Args:
        labels_raw (np.ndarray): raw labels array
        conf_raw (np.ndarray): raw confidences array

    Returns:
        tuple[list[list[int]], list[np.ndarray]]: list of labels and list of confidences
        for this labels
    """
    result_labels = []
    result_confidences = []
    for label, conf in zip(labels_raw, conf_raw):
        result_one_labels = []
        result_one_confidences = []
        for length, group in itertools.groupby(zip(label, conf), operator.itemgetter(0)):
            if length > 0:
                result_one_labels.append(length)
                result_one_confidences.append(max(list(zip(*group))[1]))
        result_labels.append(result_one_labels)
        result_confidences.append(np.array(result_one_confidences))

    return result_labels, result_confidences


def labels_to_strings(labels: list[list[int]], vocab: str) -> list[str]:
    """Converts list of labels to list of decoded strings

    Args:
        labels (list[list[int]]): list of labels
        vocab (str): vocabulary for mapping labels

    Returns:
        list[str]: list of decoded strings
    """
    strings = []
    for single_str_labels in labels:
        try:
            output_str = ''.join(
                vocab[char_index - 1]
                if char_index > 0 else '_' for char_index in single_str_labels
            )
            strings.append(output_str)
        except IndexError:
            strings.append('Error')
    return strings
