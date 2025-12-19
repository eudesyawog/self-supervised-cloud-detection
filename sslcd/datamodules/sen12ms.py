import os
import sys
import torch

import numpy as np
import albumentations as A
from argparse import ArgumentParser

import pytorch_lightning as pl

from typing import Any, Callable, Dict, Optional
from torchtyping import TensorType
# from torchgeo.datamodules.geo import NonGeoDataModule
from torchgeo.datamodules.utils import group_shuffle_split
from torchgeo.datasets import SEN12MS as BaseSEN12MS
from torch.utils.data import DataLoader, Subset
from ..transforms.moco_transforms import NullTransform

class SEN12MSDataset(BaseSEN12MS):
    def __init__(self,
                 root, 
                 split, 
                 bands, 
                 band_set = "all",
                 transforms=None, 
                 checksum=False,
                 output_packer=None,
                 *args: Any,
                 **kwargs: Any):
        super().__init__(root, split, bands, transforms, checksum)

        self.output_packer = output_packer
        self.band_set = band_set

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)

        if not self.output_packer is None:
            sample["image"] = sample["image"].numpy().astype(np.float32)
            sample["mask"] = sample["mask"].unsqueeze(0).numpy().astype(np.float32)

            split_idx = 2 if self.band_set in ["all", "s1"] else 0
            s1_img = sample["image"][:split_idx]
            s2_img = sample["image"][split_idx:]

            # print(s1_img.shape, s2_img.shape)

            if s1_img.size == 0 :
                s1_img = None 
            if s2_img.size == 0 :  
                s2_img = None 

            if "mask" in sample:
                lc_img = sample["mask"]
                if lc_img.size == 0 :  
                    lc_img = None 

            return self.output_packer(s1=s1_img, s2=s2_img, lc=lc_img)
        else:
            return sample

class SEN12MSDataModule(pl.LightningDataModule):
    name = "SEN12MS"

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        band_set: str = "all",
        data_dir: str = "./Datasets/SEN12MS",
        **kwargs: Any,
    ) -> None:
        """Initialize a new SEN12MSDataModule instance based on torchgeo code.
        https://torchgeo.readthedocs.io/en/latest/_modules/torchgeo/datamodules/sen12ms.html#SEN12MSDataModule

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            band_set: Subset of S1/S2 bands to use. Options are: "all",
                "s1", "s2-all", and "s2-reduced" where the "s2-reduced" set includes:
                B2, B3, B4, B8, B11, and B12.
            **kwargs: Additional keyword arguments passed to
                :class:`SEN12MSDataset defined at top`.
        """
        
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        
        kwargs['band_set'] = band_set
        kwargs['bands'] = SEN12MSDataset.BAND_SETS[band_set]
        kwargs['root'] = data_dir
        self.kwargs = kwargs
        print (self.kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit', 'validate']:
            season_to_int = {'winter': 0, 'spring': 1000, 'summer': 2000, 'fall': 3000}

            self.dataset = SEN12MSDataset(split='train', **self.kwargs)

            # A patch is a filename like:
            #     "ROIs{num}_{season}_s2_{scene_id}_p{patch_id}.tif"
            # This patch will belong to the scene that is uniquely identified by its
            # (season, scene_id) tuple. Because the largest scene_id is 149, we can
            # simply give each season a large number and representing a unique_scene_id
            # as (season_id + scene_id).
            scenes = []
            for scene_fn in self.dataset.ids:
                parts = scene_fn.split('_')
                season_id = season_to_int[parts[1]]
                scene_id = int(parts[3])
                scenes.append(season_id + scene_id)

            train_indices, val_indices = group_shuffle_split(
                scenes, test_size=0.2, random_state=0
            )

            self.train_dataset = Subset(self.dataset, train_indices)
            self.val_dataset = Subset(self.dataset, val_indices)
            
        if stage in ['test']:
            self.test_dataset = SEN12MSDataset(split='test',**self.kwargs)

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training.

        Returns:
            training data loader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation.

        Returns:
            validation data loader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing.

        Returns:
            testing data loader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
        )

    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--band_set", type=str, default="s2-all", choices=["all", "s1", "s2-all", "s2-reduced"])
        # parser.add_argument("--batch_size", type=int, default=64)
        # parser.add_argument("--num_workers", type=int, default=8)
        # parser.add_argument("--data_dir", type=str, default="./Datasets/SEN12MS")
        parser.add_argument("--seed", type=int, default=42)
        
        
        return parser