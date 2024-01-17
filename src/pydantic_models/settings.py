from pydantic import BaseModel
from omegaconf import OmegaConf


class DetectorSettings(BaseModel):
    onnx_weights_path: str
    iou_threshold: float
    conf_threshold: float


class RecognizerSettings(BaseModel):
    onnx_weights_path: str


class VisualizerSettings(BaseModel):
    box_color: tuple[int, int, int]
    box_thickness: int
    text_color: tuple[int, int, int]
    text_font_scale: float
    text_thickness: int
    text_y_shift: int


class AppSettings(BaseModel):
    host: str
    port: int


class Settings(BaseModel):
    barcode_detector: DetectorSettings
    barcode_recognizer: RecognizerSettings
    barcode_visualizer: VisualizerSettings
    app_settings: AppSettings
    extra_crop_val: int = 10

    @classmethod
    def from_yaml(cls, path: str) -> 'Settings':
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
