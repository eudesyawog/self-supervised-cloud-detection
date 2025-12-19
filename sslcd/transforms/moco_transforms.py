import random
from matplotlib import image
import torch

import albumentations as A
import numpy as np

from typing import Any, Optional
from torchtyping import TensorType
from albumentations.pytorch.transforms import ToTensorV2

# Because Augmentation libraries expect numpy ordering by TorchGeo Datasets provide Tensors.
def to_albumentations(img):
    if img is None:
        return None

    if len(img.shape) == 3:
        img = np.einsum("chw->hwc", img)

    if isinstance(img, torch.TensorType):
        return img.numpy()
    else:
        return img
    
class Sentinel2Normalize(A.ImageOnlyTransform):
    """Normalize a Sentinel-2 image tensor image by dividing its 16bit DN by the scale factor 10,000 and rescaling
    the tensor to the range [0,1]
    This transform does not support PIL Image.

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.
    """

    def __init__(self):
        super().__init__(True, 1)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        boa_reflectance = img.astype(np.float32).clip(0, 10000) / 10000
        return boa_reflectance
    
class MocoSEN12MSPacker:
    def __init__(
        self,
        bands: str = "all",
        patch_sz: int = 256,
        cache_size: int = 100,
    ) -> None:
        self.patch_sz = patch_sz
        self.bands = bands
        self.cache_size = cache_size

        larger_sz = min(256, int(patch_sz * 1.2))
        # These transforms are used to cut the patch from the larger image patch
        self.patch_selection = A.Compose(
            [
                A.RandomCrop(larger_sz, larger_sz, p=1),
            ],
            additional_targets={"s1": "image", "s2": "image"}
        )

        # These transforms transform the geospatial nature of the patch
        self.geo_transform = A.Compose(
            [
                A.RandomCrop(patch_sz, patch_sz, p=1),
                A.VerticalFlip(p=0.25),
                A.HorizontalFlip(p=0.25),
            ],
            additional_targets={"s1": "image", "s2": "image"},
        )

        self.radio_transform = A.Compose(
            [
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                Sentinel2Normalize(),
                ToTensorV2(),
            ]
        )

    def __call__(
        self,
        s1: Optional[TensorType] = None,
        s2: Optional[TensorType] = None,
        lc: Optional[TensorType] = None,
    ) -> Any:
        sample = {}

        # This image augmentation library is a pain to work with
        s1 = to_albumentations(s1)
        s2 = to_albumentations(s2)
        lc = to_albumentations(lc)

        images = {"image": lc}

        if self.bands in ["s1", "all"]:
            images["s1"] = s1

        if self.bands in ["all", "s2-all", "s2-reduced"]:
            images["s2"] = s2

        patches = self.patch_selection(**images)

        # Apply the geo transforms
        augmented_q = self.geo_transform(**patches)
        augmented_k = self.geo_transform(**patches)

        # Apply the radiometric transforms
        if self.bands in ["all", "s2-all", "s2-reduced"]:
            augmented_q["s2"] = self.radio_transform(image=augmented_q["s2"])["image"]
            augmented_k["s2"] = self.radio_transform(image=augmented_k["s2"])["image"]

        if self.bands == "all":
            return (augmented_q["s1"], augmented_k["s1"], augmented_q["s2"], augmented_k["s2"]), augmented_q["image"]
        elif self.bands == "s1":
            return (augmented_q["s1"], augmented_k["s1"]), augmented_q["image"]
        else:
            return (augmented_q["s2"], augmented_k["s2"]), augmented_q["image"]
        