import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
import rasterio
from torch.utils.data import Dataset
from scipy.ndimage import zoom

class WHUS2CDDataset(Dataset):

    WHUS2_CD_REMAP = (10, 0, 1, 2, 4, 5, 6, 3, 7, 11, 12, 8, 9)

    def __init__(
        self,
        root: str,
        phase: str = "train",
        bin_size_for_stratification: int = 10,
        subset_fraction: float = 1.0,
        patch_size: int = 256,
        pretraining: bool = False,
        *args,
        **kwargs
    ):
        self.root = root
        self.phase = phase
        self.patch_size = patch_size
        self.pretraining = pretraining

        self.df = pd.read_csv(os.path.join(self.root, f"{phase}_set.csv"))
        self.df["klass"] = self.df.cloudiness.astype(int) // bin_size_for_stratification

        # Reduce the dataset to a smaller balanced version to see how the model performs

        if subset_fraction < 1.0:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - subset_fraction, random_state=42)
            train_idx, _ = next(sss.split(self.df, self.df.klass))
            self.df = self.df.iloc[train_idx]

    def __len__(self):
        return len(self.df)
    
    def read_img_and_labels (self, item):
        
        path_lbl = os.path.join(self.root, item.lbl_path)
        path_10m = path_lbl.replace("labels", "10m").replace("_Mask.tif",".tif")
        path_20m = path_lbl.replace("labels", "20m").replace("_Mask.tif",".tif")
        path_60m = path_lbl.replace("labels", "60m").replace("_Mask.tif",".tif")

        with rasterio.open(path_lbl) as base_src:
            lbl = base_src.read()

            imgs = []
            with rasterio.open(path_10m) as src:
                imgs.append(src.read())

            with rasterio.open(path_20m) as src:
                im = zoom(src.read(), (1, 2, 2), order=0, mode="nearest", grid_mode=True)
                imgs.append(im)

            with rasterio.open(path_60m) as src:
                im = zoom(src.read(), (1, 6, 6), order=0, mode="nearest", grid_mode=True)
                imgs.append(im)

        img = np.concatenate(imgs, axis=0)[
            self.WHUS2_CD_REMAP,
        ]

        img = img.astype(np.float32) / 10000
        lbl = lbl.astype(np.int32) // 255
        
        return img, lbl

    def __getitem__(self, index):

        item = self.df.iloc[index]

        img, lbl = self.read_img_and_labels(item)

        if self.pretraining:
            # find another sample of the same class 
            samples_idx = self.df.loc[self.df["klass"]==item.klass].index.values
            samples_idx = samples_idx[samples_idx!=index] 
            i = np.random.choice(samples_idx, size=1, replace=False)[0]
            img2, _ = self.read_img_and_labels(self.df.iloc[i])

            return (
               [torch.from_numpy(img).float(), torch.from_numpy(img2).float()],
               torch.from_numpy(np.array([item.klass])).long()[0] 
            )

        else:
            h, w = img.shape[1:]

            x = 0 if w==self.patch_size else np.random.randint(0, w - self.patch_size)
            y = 0 if h==self.patch_size else np.random.randint(0, h - self.patch_size)

            i_c = img[:, y : y + self.patch_size, x : x + self.patch_size]
            l_c = lbl[:, y : y + self.patch_size, x : x + self.patch_size]

            return torch.from_numpy(i_c).float(), torch.from_numpy(l_c).float()

class CloudSEN12Dataset(Dataset):
    def __init__(
        self,
        root: str,
        phase: str = "train",
        bin_size_for_stratification: int = 10,
        subset_fraction: float = 1.0,
        patch_size: int = 256,
        pretraining: bool = False,
        task: str = "cloud",
        *args,
        **kwargs,
    ):
        self.root = root
        self.phase = phase
        self.patch_size = patch_size
        self.pretraining = pretraining
        self.task = task

        self.df = pd.read_csv(os.path.join(self.root, f"{phase}_set.csv"))
        self.df["klass"] = self.df.cloudiness.astype(int) // bin_size_for_stratification

        # Reduce the dataset to a smaller balanced version to see how the model performs

        if subset_fraction < 1.0:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - subset_fraction, random_state=42)
            train_idx, _ = next(sss.split(self.df, self.df.klass))
            self.df = self.df.iloc[train_idx]

    def __len__(self):
        return len(self.df)
    
    def read_img_and_labels (self, item):
        path_lbl = os.path.join(self.root, item.lbl_path)
        path_img = os.path.join(self.root, item.img_path)
        
        with rasterio.open(path_lbl) as base_src:
            lbl = base_src.read()
            if self.task == "cloud":
                lbl = np.where((lbl==1)|(lbl==2),1,0)
            elif self.task == "shadow":
                lbl = np.where((lbl==3),1,0)
            elif self.task == "joint":
                lbl = np.where((lbl==1)|(lbl==2)|(lbl==3),1,0)
            elif self.task == "multi":
                pass

        with rasterio.open(path_img) as src:
            assert src.count == 13
            img = src.read()

        img = img.astype(np.float32) / 10000
        lbl = lbl.astype(np.uint8)

        # Pad to have patch size equal to 512x512
        img_padded = np.zeros((img.shape[0],512,512),dtype = np.float32)
        img_padded[:,:img.shape[1],:img.shape[2]] = img
        img = img_padded

        lbl_padded = np.zeros((lbl.shape[0],512,512),dtype = np.uint8)
        lbl_padded[:,:lbl.shape[1],:lbl.shape[2]] = lbl
        lbl = lbl_padded

        return img, lbl
    
    def __getitem__(self, index):
        item = self.df.iloc[index]
        img, lbl = self.read_img_and_labels(item)

        if self.pretraining:
            # find another sample of the same class 
            samples_idx = self.df.loc[self.df["klass"]==item.klass].index.values
            samples_idx = samples_idx[samples_idx!=index] 
            i = np.random.choice(samples_idx, size=1, replace=False)[0]
            img2, _ = self.read_img_and_labels(self.df.iloc[i])

            return (
               [torch.from_numpy(img).float(), torch.from_numpy(img2).float()],
               torch.from_numpy(np.array([item.klass])).long()[0] 
            )
        else:
            h, w = img.shape[1:]

            x = 0 if w==self.patch_size else np.random.randint(0, w - self.patch_size)
            y = 0 if h==self.patch_size else np.random.randint(0, h - self.patch_size)

            i_c = img[:, y : y + self.patch_size, x : x + self.patch_size]
            l_c = lbl[:, y : y + self.patch_size, x : x + self.patch_size]

            return torch.from_numpy(i_c).float(), torch.from_numpy(l_c).float()
