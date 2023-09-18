from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from enum import EnumMeta
from time import time

import torch.utils.data as data

import os
import re
import torch
import tarfile
from PIL import Image
import cv2
import numpy as np
import json
import pandas as pd
from tqdm import tqdm

IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp']


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def find_images_and_targets(folder, types=IMG_EXTENSIONS, class_to_idx=None, sort=True):
    labels = []
    filenames = []
    for root, subdirs, files in os.walk(folder, topdown=False):
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in types:
                filenames.append(os.path.join(root, f))
                if len(f.split("_")) > 1:
                    labels.append(f.split("_")[1])
                else:
                    labels.append("other")
    
    if class_to_idx is None:
        # building class index
        unique_labels = set(labels)
        sorted_labels = list(sorted(unique_labels, key=natural_key))
        class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    images_and_targets = zip(filenames, [class_to_idx[l] for l in labels])

    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    return images_and_targets, class_to_idx, labels

class ODDDataset(data.Dataset):
    def __init__(
            self,
            root,
            transform=None,
            load_type="pil"
            ):

        assert load_type in ["pil", "cv2"], "error load type"
        
        class_to_idx = None
        images, class_to_idx, labels = find_images_and_targets(root, class_to_idx=class_to_idx)
        class_to_idx2 = {idx: c for idx, c in enumerate(class_to_idx)}
        
        if len(images) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))
        self.root = root
        self.samples = images
        self.imgs = self.samples  # torchvision ImageFolder compat
        self.class_to_idx = class_to_idx
        self.load_type = load_type
        self.preload = False
        self.transform = transform
        self.labels = labels

    def __getitem__(self, index):
        path, target = self.samples[index]
        if not self.preload:
            if self.load_type == "pil":
                pil_img = Image.open(path).convert('RGB')
                img = np.asarray(pil_img, dtype=np.uint8)
            else:
                im_bgr = cv2.imread(path)
                img = im_bgr[:, :, ::-1]
        else:
            img = self.imgs[index]
            
        if self.transform is not None:
            img = self.transform(image=img)['image']
        if target is None:
            target = torch.zeros(1).long()
        return img, target

    def __len__(self):
        return len(self.imgs)

    def filenames(self, indices=[], basename=False):
        if indices:
            if basename:
                return [os.path.basename(self.samples[i][0]) for i in indices]
            else:
                return [self.samples[i][0] for i in indices]
        else:
            if basename:
                return [os.path.basename(x[0]) for x in self.samples]
            else:
                return [x[0] for x in self.samples]

class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits
        self.labels = dataset.labels

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)