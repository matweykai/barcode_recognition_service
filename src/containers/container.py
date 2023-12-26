from dependency_injector import containers, providers

from src.services.detector import BarcodeDetector
from src.services.recognizer import BarcodeRecognizer
from src.services.analyzer import BarcodeAnalyzer


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
