"""NIfTI image loader for medical imaging datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


class NiftiLoader:
    """Load NIfTI (``.nii`` / ``.nii.gz``) volumes into NumPy arrays.

    Parameters
    ----------
    path:
        Path to a NIfTI file.
    normalize:
        When ``True``, voxel values are scaled to ``[0, 1]``.
    """

    def __init__(self, path: str | Path, normalize: bool = True) -> None:
        self.path = Path(path)
        self.normalize = normalize

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> Tuple[np.ndarray, dict]:
        """Return ``(volume_array, metadata)``.

        The returned array has shape ``(X, Y, Z)`` or ``(X, Y, Z, T)`` for
        4-D time-series data, matching the NIfTI voxel ordering.
        """
        import nibabel as nib  # type: ignore

        img = nib.load(str(self.path))
        data = np.asarray(img.dataobj, dtype=np.float32)
        if self.normalize:
            data = self._normalize(data)
        header = img.header
        meta = {
            "shape": data.shape,
            "affine": img.affine.tolist(),
            "voxel_sizes": header.get_zooms(),
            "data_dtype": str(header.get_data_dtype()),
        }
        return data, meta

    def get_slice(self, axis: int, index: int) -> np.ndarray:
        """Extract a 2-D slice along *axis* at position *index*.

        Parameters
        ----------
        axis:
            Axis to slice along – ``0`` (sagittal), ``1`` (coronal), or
            ``2`` (axial).
        index:
            Slice index along the chosen axis.
        """
        volume, _ = self.load()
        if axis == 0:
            return volume[index, :, :]
        elif axis == 1:
            return volume[:, index, :]
        elif axis == 2:
            return volume[:, :, index]
        raise ValueError(f"axis must be 0, 1 or 2; got {axis}.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(array: np.ndarray) -> np.ndarray:
        arr_min, arr_max = array.min(), array.max()
        if arr_max > arr_min:
            return (array - arr_min) / (arr_max - arr_min)
        return array
