from dependency_injector import containers, providers

from src.services.detector import BarcodeDetector
from src.services.recognizer import BarcodeRecognizer
from src.services.analyzer import BarcodeAnalyzer
from src.services.visualizer import BarcodesVisualizer


class AppContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    detector = providers.Singleton(
        BarcodeDetector,
        onnx_weights_path=config.barcode_detector.onnx_weights_path,
        iou_threshold=config.barcode_detector.iou_threshold,
        conf_threshold=config.barcode_detector.conf_threshold,
    )

    recognizer = providers.Singleton(
        BarcodeRecognizer,
        onnx_weights_path=config.barcode_recognizer.onnx_weights_path,
    )

    barcode_analyzer = providers.Singleton(
        BarcodeAnalyzer,
        detector=detector,
        recognizer=recognizer,
        extra_crop_val=config.extra_crop_val,
    )

    barcode_visualizer = providers.Factory(
        BarcodesVisualizer,
        box_color=config.barcode_visualizer.box_color,
        box_thickness=config.barcode_visualizer.box_thickness,
        text_color=config.barcode_visualizer.text_color,
        text_font_scale=config.barcode_visualizer.text_font_scale,
        text_thickness=config.barcode_visualizer.text_thickness,
        text_y_shift=config.barcode_visualizer.text_y_shift,
    )
