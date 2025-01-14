import os
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class MZIrisDataset(Dataset):
    def __init__(self, data_path, image_dir) -> None:
        self.data = pd.read_csv(data_path)
        self.img_dir = image_dir

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx):
        fn_left = str(self.data["sequenceid_left"][idx])
        fn_right = str(self.data["sequenceid_right"][idx])
        if fn_left[-5:] != ".tiff":
            fn_left = fn_left + ".tiff"
        if fn_right[-5:] != ".tiff":
            fn_right = fn_right + ".tiff"

        sid_left = fn_left[:-5]
        sid_right = fn_right[:-5]

        fn_left = os.path.join(self.img_dir, fn_left)
        fn_right = os.path.join(self.img_dir, fn_right)
        img_left = Image.open(fn_left).convert("L")
        img_right = Image.open(fn_right).convert("L")

        img_left = pil_to_tensor(img_left)
        img_right = pil_to_tensor(img_right)

        img_left = img_left.float() / 255
        img_right = img_right.float() / 255

        label = self.data["label"][idx]

        return sid_left, sid_right, img_left, img_right, label


class MZIrisDatasetMask(Dataset):
    def __init__(self, data_path, image_dir, hmap_dir, inverse) -> None:
        self.data = pd.read_csv(data_path)
        self.img_dir = image_dir
        self.hmap_dir = hmap_dir
        self.inverse = inverse

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx):
        fn_left = str(self.data["sequenceid_left"][idx])
        fn_right = str(self.data["sequenceid_right"][idx])
        if fn_left[-5:] != ".tiff":
            fn_left = fn_left + ".tiff"
        if fn_right[-5:] != ".tiff":
            fn_right = fn_right + ".tiff"

        sid_left = fn_left[:-5]
        sid_right = fn_right[:-5]

        img_path_left = os.path.join(self.img_dir, fn_left)
        img_path_right = os.path.join(self.img_dir, fn_right)
        img_left = Image.open(img_path_left).convert("L")
        img_right = Image.open(img_path_right).convert("L")
        img_left = pil_to_tensor(img_left)
        img_right = pil_to_tensor(img_right)
        img_left = img_left.float() / 255
        img_right = img_right.float() / 255

        hmap_path_left = os.path.join(self.hmap_dir, fn_left[:-5] + "_seg_mask.png")
        hmap_path_right = os.path.join(self.hmap_dir, fn_right[:-5] + "_seg_mask.png")
        hmap_left = Image.open(hmap_path_left).convert("L")
        hmap_right = Image.open(hmap_path_right).convert("L")
        hmap_left = pil_to_tensor(hmap_left)
        hmap_right = pil_to_tensor(hmap_right)
        # hmap_left = torch.squeeze(resize(hmap_left, (15, 20)))
        # hmap_right = torch.squeeze(resize(hmap_right, (15, 20)))
        hmap_left = hmap_left / 255  # normalization
        hmap_right = hmap_right / 255  # normalization
        if self.inverse:
            hmap_left = 1 - hmap_left
            hmap_right = 1 - hmap_right

        # pair-wise multiplication
        img_left = torch.mul(img_left, hmap_left)
        img_right = torch.mul(img_right, hmap_right)

        label = self.data["label"][idx]

        return sid_left, sid_right, img_left, img_right, label
