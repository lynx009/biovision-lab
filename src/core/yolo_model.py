"""YOLO-based detector wrapper for medical imaging (YOLOv8/v10, RT-DETR)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class YOLODetector:
    """Thin wrapper around Ultralytics YOLO / RT-DETR for medical detection.

    Parameters
    ----------
    model_variant:
        One of ``"yolov8n"``, ``"yolov8s"``, ``"yolov8m"``, ``"yolov8l"``,
        ``"yolov8x"``, ``"yolov10n"`` … ``"yolov10x"``, or ``"rtdetr-l"`` /
        ``"rtdetr-x"``.
    num_classes:
        Number of target classes (excluding background).
    pretrained:
        Whether to load Ultralytics pretrained weights as a starting point.
    """

    def __init__(
        self,
        model_variant: str = "yolov8n",
        num_classes: int = 2,
        pretrained: bool = True,
    ) -> None:
        self.model_variant = model_variant
        self.num_classes = num_classes
        self.pretrained = pretrained
        self._model = None

    # ------------------------------------------------------------------
    # Lazy model initialisation (avoids import errors when ultralytics
    # is not installed in the current environment).
    # ------------------------------------------------------------------

    def _load(self) -> None:
        from ultralytics import YOLO  # type: ignore

        weight = f"{self.model_variant}.pt" if self.pretrained else f"{self.model_variant}.yaml"
        self._model = YOLO(weight)

    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        batch: int = 8,
        imgsz: int = 512,
        project: str = "experiments",
        name: str = "yolo_run",
        **kwargs,
    ) -> None:
        """Train the detector on a YOLO-format dataset."""
        if self._model is None:
            self._load()
        self._model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            project=project,
            name=name,
            **kwargs,
        )

    def predict(self, source, **kwargs):
        """Run inference on *source* (path, numpy array, or URL)."""
        if self._model is None:
            self._load()
        return self._model.predict(source, **kwargs)

    def export(self, format: str = "onnx", export_dir: str = "deployments") -> Path:
        """Export the model to *format* and move it to *export_dir*."""
        if self._model is None:
            self._load()
        result = self._model.export(format=format)
        dest = Path(export_dir) / Path(result).name
        Path(export_dir).mkdir(parents=True, exist_ok=True)
        Path(result).rename(dest)
        return dest
