import json
import os
from collections import defaultdict
import random
import math

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
import torchvision.transforms as transforms

class TETRO(Dataset):
    def __init__(
        self,
        data_dir: str = "data/tetrominoes",
        img_size: int = 64,
        crop_size: int = 80,
        padding_size: int = 28,
        train: bool = True,
    ):
        super().__init__()

        self.img_size = img_size
        self.crop_size = crop_size
        self.train = train
        self.stage = "train" if train else "val"

        self.image_dir = os.path.join(data_dir, "images", self.stage)
        self.mask_dir = os.path.join(data_dir, "masks", self.stage)
        self.scene_dir = os.path.join(data_dir, "scenes")
        self.metadata = json.load(
            open(os.path.join(self.scene_dir, f"TETROMINOES_{self.stage}_scenes.json"))
        )

        self.files = sorted(os.listdir(self.image_dir))
        self.num_files = len(self.files)

        self.transform = transforms.Compose([
            transforms.Pad(padding=(padding_size,padding_size), fill=1, padding_mode='constant'),
            transforms.CenterCrop(crop_size),
            transforms.Resize((img_size, img_size)),
        ])

    def __getitem__(self, index):
        filename = self.metadata["scenes"][index]["image_filename"]
        img = (
            read_image(os.path.join(self.image_dir, filename), ImageReadMode.RGB)
            .float()
            .div(255.0)
        )
        img = self.transform(img)
            
        return img

    def __len__(self):
        return self.num_files
    
class PTR(Dataset):
    def __init__(
        self,
        data_dir: str = "data/PTR_192",
        img_size: int = 128,
        crop_size: int = 192,
        train: bool = True,
    ):
        super().__init__()

        self.img_size = img_size
        self.train = train
        self.stage = "train" if train else "val"

        self.image_dir = os.path.join(data_dir, "images", self.stage)

        self.files = sorted(os.listdir(self.image_dir))
        self.num_files = len(self.files)

        self.transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Resize((img_size, img_size)),
        ])

    def __getitem__(self, index):
        image_filename = self.files[index]
        img = (
            read_image(os.path.join(self.image_dir, image_filename), ImageReadMode.RGB)
            .float()
            .div(255.0)
        )
        img = self.transform(img)

        return img

    def __len__(self):
        return self.num_files

class ClevrTex10(Dataset):
    def __init__(
        self,
        data_dir: str = "data/clevrtex",
        img_size: int = 128,
        crop_size: int = 192,
        train: bool = True,
    ):
        super().__init__()

        self.img_size = img_size
        self.train = train
        self.stage = "train" if train else "val"

        self.image_dir = os.path.join(data_dir, "images", self.stage)

        self.files = sorted(os.listdir(self.image_dir))
        self.num_files = len(self.files)

        self.transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Resize((img_size, img_size)),
        ])

    def __getitem__(self, index):
        image_filename = self.files[index]
        img = (
            read_image(os.path.join(self.image_dir, image_filename), ImageReadMode.RGB)
            .float()
            .div(255.0)
        )
        img = self.transform(img)

        return img

    def __len__(self):
        return self.num_files
    
class ClevrTex6(Dataset):
    def __init__(
        self,
        data_dir: str = "data/clevrtex6",
        img_size: int = 128,
        crop_size: int = 192,
        train: bool = True,
    ):
        super().__init__()

        self.img_size = img_size
        self.train = train
        self.stage = "train" if train else "val"

        self.image_dir = os.path.join(data_dir, "images", self.stage)

        self.files = sorted(os.listdir(self.image_dir))
        self.num_files = len(self.files)

        self.transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Resize((img_size, img_size)),
        ])

    def __getitem__(self, index):
        image_filename = self.files[index]
        img = (
            read_image(os.path.join(self.image_dir, image_filename), ImageReadMode.RGB)
            .float()
            .div(255.0)
        )
        img = self.transform(img)

        return img

    def __len__(self):
        return self.num_files