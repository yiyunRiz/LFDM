"""Semantic segmentation datasets used by LFDM."""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import data.util as Util


class SegmentationDataset(Dataset):
    """Load aligned remote-sensing images and integer semantic masks."""

    def __init__(
        self,
        image_root,
        label_root=None,
        resolution=256,
        split="train",
        data_len=-1,
        label_offset=0,
        require_labels=True,
    ):
        self.image_paths = Util.get_paths_from_images(image_root)
        self.label_root = label_root
        self.resolution = resolution
        self.split = split
        self.label_offset = label_offset
        self.require_labels = require_labels
        if require_labels and not label_root:
            raise ValueError("A labelroot is required for supervised training/evaluation.")

        if label_root:
            labels_by_stem = {
                Path(path).stem: path for path in Util.get_paths_from_images(label_root)
            }
            self.label_paths = []
            for image_path in self.image_paths:
                stem = Path(image_path).stem
                if stem not in labels_by_stem:
                    raise FileNotFoundError(
                        "No mask with stem '{}' was found in '{}'.".format(stem, label_root)
                    )
                self.label_paths.append(labels_by_stem[stem])
        else:
            self.label_paths = None

        dataset_len = len(self.image_paths)
        self.data_len = dataset_len if data_len <= 0 else min(data_len, dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.resolution, self.resolution), Image.BICUBIC)
        image = Util.transform_augment_cd(image, split=self.split, min_max=(-1, 1))
        sample = {"img": image, "name": os.path.basename(image_path)}

        if self.label_paths is not None:
            label_image = Image.open(self.label_paths[index]).resize(
                (self.resolution, self.resolution), Image.NEAREST
            )
            label = np.array(label_image, dtype=np.int64)
            if label.ndim == 3:
                label = label[..., 0]
            label = label - self.label_offset
            sample["label"] = torch.from_numpy(label).long()
        return sample
