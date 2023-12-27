from pydantic import BaseModel
from omegaconf import OmegaConf


class DetectorSettings(BaseModel):
    onnx_weights_path: str
    iou_threshold: float
    conf_threshold: float


class RecognizerSettings(BaseModel):
    onnx_weights_path: str


class AppSettings(BaseModel):
    host: str
    port: int


class Settings(BaseModel):
    barcode_detector: DetectorSettings
    barcode_recognizer: RecognizerSettings
    app_settings: AppSettings
    extra_crop_val: int = 10

    @classmethod
    def from_yaml(cls, path: str) -> 'Settings':
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
