from pydantic import BaseModel
from omegaconf import OmegaConf


class DetectorSettings(BaseModel):
    onnx_weights_path: str
    iou_threshold: float
    conf_threshold: float


class RecognizerSettings(BaseModel):
    onnx_weights_path: str


class Settings:
    barcode_detector: DetectorSettings
    barcode_recognizer: RecognizerSettings
    extra_crop_val: int = 10

    @classmethod
    def from_yaml(cls, path: str) -> 'Settings':
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
