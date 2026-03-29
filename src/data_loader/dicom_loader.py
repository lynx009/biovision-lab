"""DICOM image loader for medical imaging datasets."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np


class DicomLoader:
    """Load DICOM series or single slices into NumPy arrays.

    Parameters
    ----------
    path:
        Path to a ``.dcm`` file **or** a directory that contains a DICOM
        series.
    normalize:
        When ``True``, pixel values are scaled to ``[0, 1]``.
    """

    def __init__(self, path: str | Path, normalize: bool = True) -> None:
        self.path = Path(path)
        self.normalize = normalize

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> Tuple[np.ndarray, dict]:
        """Return ``(image_array, metadata)`` for the given path.

        * If *path* is a single ``.dcm`` file the returned array has shape
          ``(H, W)`` or ``(H, W, C)``.
        * If *path* is a directory the full 3-D volume is reconstructed and
          the array has shape ``(slices, H, W)``.
        """
        if self.path.is_dir():
            return self._load_series(self.path)
        return self._load_single(self.path)

    def get_slice(self, index: int) -> np.ndarray:
        """Return a single 2-D slice from a DICOM series directory."""
        slices = self._sorted_dcm_files(self.path)
        if index < 0 or index >= len(slices):
            raise IndexError(f"Slice index {index} out of range (0–{len(slices) - 1}).")
        import pydicom  # type: ignore

        ds = pydicom.dcmread(slices[index])
        return self._to_float(ds.pixel_array)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_single(self, dcm_path: Path) -> Tuple[np.ndarray, dict]:
        import pydicom  # type: ignore

        ds = pydicom.dcmread(str(dcm_path))
        image = self._to_float(ds.pixel_array)
        meta = {
            "PatientID": getattr(ds, "PatientID", ""),
            "Modality": getattr(ds, "Modality", ""),
            "SliceThickness": getattr(ds, "SliceThickness", None),
            "PixelSpacing": getattr(ds, "PixelSpacing", None),
        }
        return image, meta

    def _load_series(self, series_dir: Path) -> Tuple[np.ndarray, dict]:
        import pydicom  # type: ignore

        files = self._sorted_dcm_files(series_dir)
        slices = [pydicom.dcmread(str(f)) for f in files]
        volume = np.stack([self._to_float(s.pixel_array) for s in slices], axis=0)
        first = slices[0]
        meta = {
            "PatientID": getattr(first, "PatientID", ""),
            "Modality": getattr(first, "Modality", ""),
            "SliceThickness": getattr(first, "SliceThickness", None),
            "PixelSpacing": getattr(first, "PixelSpacing", None),
            "num_slices": len(slices),
        }
        return volume, meta

    @staticmethod
    def _sorted_dcm_files(directory: Path) -> List[Path]:
        import pydicom  # type: ignore

        files = sorted(directory.glob("*.dcm"))
        if not files:
            raise FileNotFoundError(f"No .dcm files found in {directory}")

        def _instance_number(p: Path) -> int:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
            return int(getattr(ds, "InstanceNumber", 0))

        return sorted(files, key=_instance_number)

    def _to_float(self, array: np.ndarray) -> np.ndarray:
        arr = array.astype(np.float32)
        if self.normalize:
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max > arr_min:
                arr = (arr - arr_min) / (arr_max - arr_min)
        return arr
