"""U-Net segmentation model backed by MONAI for medical imaging."""

from __future__ import annotations

from typing import Sequence


class UNetSegmentor:
    """MONAI U-Net wrapper for volumetric / 2-D medical image segmentation.

    Parameters
    ----------
    spatial_dims:
        Number of spatial dimensions – ``2`` for 2-D slices, ``3`` for
        volumetric data.
    in_channels:
        Number of input channels (1 for greyscale MRI, 3 for RGB).
    out_channels:
        Number of output segmentation classes.
    features:
        Feature map sizes at each encoder stage.
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (16, 32, 64, 128, 256, 16),
    ) -> None:
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = tuple(features)
        self._model = None

    # ------------------------------------------------------------------
    # Lazy build so MONAI is only required when actually used.
    # ------------------------------------------------------------------

    def build(self):
        """Instantiate the underlying MONAI BasicUNet."""
        from monai.networks.nets import BasicUNet  # type: ignore

        self._model = BasicUNet(
            spatial_dims=self.spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            features=self.features,
        )
        return self._model

    @property
    def model(self):
        if self._model is None:
            self.build()
        return self._model

    def forward(self, x):
        return self.model(x)

    def __call__(self, x):
        return self.forward(x)
