import os
import warnings
import rasterio

import numpy as np
import pytorch_lightning as pl
import pandas as pd

from argparse import ArgumentParser
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
from typing import Any
from torch.utils.data import DataLoader

from glob import glob
from ..tools.utils import dataset_with_index
from .cd_datasets import WHUS2CDDataset, CloudSEN12Dataset

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

class CloudDetectionDataModule(pl.LightningDataModule):
    name = "CloudDetectionDataModule"

    def __init__(
        self,
        data_dir: str,
        seed: int,
        batch_size: int = 64,
        num_workers: int = 0,
        training_set_fraction: float = 0.7,
        limit_dataset: float = 1.0,
        patch_size: int = 256,
        pretraining: bool = False,
        dataset: str = "WHUS2-CD+",
        task: str = "cloud",
        label_type: str = ["high"],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_frac = training_set_fraction
        self.limit_dataset_fraction = limit_dataset
        self.patch_size = patch_size
        self.pretraining = pretraining
        self.dataset = dataset
        self.task = task
        self.label_type = label_type

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", type=str, default="./Datasets")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--training_set_fraction", type=float, default=0.7)
        parser.add_argument("--patch_size", type=int, default=256)
        parser.add_argument("--pretraining", action="store_true")
        parser.add_argument("--dataset", type=str, default="WHUS2-CD+", 
                            choices = ["WHUS2-CD+","CloudSEN12"] )
        parser.add_argument("--task", type=str, default="cloud",choices = ["cloud","shadow","joint","multi"], help="CloudSEN12 task to perform") 
        parser.add_argument("--label_type", nargs='+',choices=['high','scribble','nolabel'],default=['high'],help="CloudSEN12 dataset label type to consider" )
        
        return parser

    def _build_dataset_index_file(self, stage: str = "train"):
        
        dataset_stats = []
            
        if self.dataset == "WHUS2-CD+":
            files_lbls = glob(os.path.join(self.data_dir, stage, "labels", "*.tif"))

            for f in files_lbls:
                raster = rasterio.open(f)
                data = raster.read()
                raster.close()

                fid = os.path.splitext(os.path.basename(f))[0]
                counts = np.bincount(data.ravel(), minlength=256)
                counts = (counts / counts.sum()) * 100

                dataset_stats.append(
                    {
                        "fid": fid,
                        "lbl_path": os.path.relpath(f, self.data_dir),
                        "cloudiness": counts[255],
                        "clear": counts[128],
                        "nodata": counts[0],
                    }
                )

        elif self.dataset == "CloudSEN12":
            if stage == "train":
                is_test = 0
            elif stage == "valid":
                is_test = 1
            elif stage == "test":
                is_test = 2
            meta_df = pd.read_csv(os.path.join(self.data_dir,"cloudsen12_metadata.csv"))

            ip = meta_df.loc[(meta_df['test']==is_test)&(meta_df['label_type'].isin(self.label_type))]
            
            for _, row in ip.iterrows():
                roi = row['roi_id']
                fid = row["s2_id_gee"]
                if row['label_type'] == 'high':
                    lbl_path = os.path.join("ROI",roi,fid,"labels","manual_hq.tif")
                elif row['label_type'] == 'scribble':
                    lbl_path = os.path.join("ROI",roi,fid,"labels","manual_sc.tif")
                else:
                    lbl_path = None

                with rasterio.open(os.path.join(self.data_dir,lbl_path)) as ds:
                    data = ds.read(1)
                    data = np.where((data==1)|(data==2),1,0)
                    counts = np.bincount(data.ravel(),minlength=2)
                    counts = (counts / counts.sum()) * 100
            
                dataset_stats.append(
                    {
                        "roi": roi,
                        "fid": fid,
                        "lbl_path": lbl_path,
                        "img_path": os.path.join("ROI",roi,fid,"S2L1C.tif"),
                        "cloudiness": counts[1],
                        "clear": counts[0],
                    }
                )

        return pd.DataFrame(dataset_stats)

    def prepare_data(self) -> None:
        if self.dataset == "WHUS2-CD+":
            for stage in ["train","test"]:
                if not os.path.exists(os.path.join(self.data_dir, f"{stage}_set.csv")):
                    df_stats = self._build_dataset_index_file(stage)
                    df_stats.to_csv(os.path.join(self.data_dir, f"{stage}_set.csv"), index=None)
        
        elif self.dataset == "CloudSEN12":
            for stage in ["train","valid","test"]:
                if not os.path.exists(os.path.join(self.data_dir, f"{stage}_set.csv")):
                    df_stats = self._build_dataset_index_file(stage)
                    df_stats.to_csv(os.path.join(self.data_dir, f"{stage}_set.csv"), index=None)
        
        else:
            raise ValueError(f'Dataset {self.dataset} not supported')
            
    def setup(self, stage="fit"):
        if stage == "fit" or stage is None:
            if self.dataset == "WHUS2-CD+":
                assert self.task == "cloud"
                self.base_dataset = WHUS2CDDataset(
                    self.data_dir,
                    "train",
                    bin_size_for_stratification=10,
                    subset_fraction=self.limit_dataset_fraction,
                    patch_size=self.patch_size,
                    pretraining=self.pretraining
                )
                print (len(self.base_dataset))

                # Split the dataset into 10 classes so we can stratify it into train and val
                classes = self.base_dataset.df.cloudiness.astype(int) // 10
                train_indices, val_indices = next(
                    StratifiedShuffleSplit(
                        train_size=self.train_frac, test_size=1 - self.train_frac, n_splits=2, random_state=self.seed
                    ).split(np.arange(len(classes)), classes)
                )

                if self.pretraining:
                    self.train_dataset = dataset_with_index(Subset)(self.base_dataset, train_indices)     
                else:
                    self.train_dataset = Subset(self.base_dataset, train_indices)

                self.val_dataset = Subset(self.base_dataset, val_indices)

                print (len(self.train_dataset), len(self.val_dataset))
            
            elif self.dataset == "CloudSEN12":

                if self.pretraining:
                    self.train_dataset = dataset_with_index(CloudSEN12Dataset)(self.data_dir,
                                                                                phase="train",
                                                                                patch_size=self.patch_size,
                                                                                pretraining=self.pretraining,
                                                                                task = self.task,
                                                                                subset_fraction=self.limit_dataset_fraction)
                else:
                    self.train_dataset = CloudSEN12Dataset(self.data_dir,
                                                            phase="train",
                                                            patch_size=self.patch_size,
                                                            pretraining=self.pretraining,
                                                            task = self.task,
                                                            subset_fraction=self.limit_dataset_fraction)
        
                self.val_dataset = CloudSEN12Dataset(
                    self.data_dir,
                    phase="valid",
                    patch_size=self.patch_size,
                    pretraining=self.pretraining,
                    task = self.task
                )

                print (len(self.train_dataset), len(self.val_dataset))
            
        elif stage == "test":
            if self.dataset == "WHUS2-CD+":
                self.test_dataset = WHUS2CDDataset( self.data_dir,
                                                    "test",
                                                    patch_size=self.patch_size)

            elif self.dataset == "CloudSEN12":
                self.test_dataset = CloudSEN12Dataset( self.data_dir,
                                                        "test",
                                                        patch_size=self.patch_size,
                                                        task = self.task)
                
    def train_dataloader(self):
        print (type(self.train_dataset))
        d = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True if self.pretraining else False,
            pin_memory=True,
        )

        print (next(iter(d)))
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True if self.pretraining else False,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

