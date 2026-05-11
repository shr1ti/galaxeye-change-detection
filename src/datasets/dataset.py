import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset


def remap_mask(mask):

    remapped = mask.copy()

    remapped[remapped == 1] = 0
    remapped[remapped == 2] = 1
    remapped[remapped == 3] = 1

    return remapped


class ChangeDetectionDataset(Dataset):

    def __init__(
        self,
        pre_images,
        post_images,
        masks,
        transform=None
    ):

        self.pre_images = pre_images
        self.post_images = post_images
        self.masks = masks

        self.transform = transform

    def __len__(self):

        return len(self.pre_images)

    def __getitem__(self, idx):

        # =========================
        # LOAD PRE-EVENT EO IMAGE
        # =========================

        pre = cv2.imread(self.pre_images[idx])

        pre = cv2.cvtColor(
            pre,
            cv2.COLOR_BGR2RGB
        )

        # =========================
        # LOAD POST-EVENT SAR IMAGE
        # =========================

        post = cv2.imread(
            self.post_images[idx],
            cv2.IMREAD_GRAYSCALE
        )

        # =========================
        # LOAD MASK
        # =========================

        mask = cv2.imread(
            self.masks[idx],
            cv2.IMREAD_GRAYSCALE
        )

        # =========================
        # REMAP MASK TO BINARY
        # =========================

        mask = remap_mask(mask)

        # =========================
        # RESIZE IMAGES + MASK
        # =========================

        pre = cv2.resize(
            pre,
            (256, 256)
        )

        post = cv2.resize(
            post,
            (256, 256)
        )

        mask = cv2.resize(
            mask,
            (256, 256),
            interpolation=cv2.INTER_NEAREST
        )

        # =========================
        # NORMALIZE IMAGES
        # =========================

        pre = pre.astype(np.float32) / 255.0

        post = post.astype(np.float32) / 255.0

        # =========================
        # ADD CHANNEL DIMENSION TO SAR
        # =========================

        post = np.expand_dims(
            post,
            axis=-1
        )

        # =========================
        # STACK EO + SAR CHANNELS
        # FINAL SHAPE → (H, W, 4)
        # =========================

        image = np.concatenate(
            [pre, post],
            axis=-1
        )

        # =========================
        # APPLY AUGMENTATIONS
        # =========================

        if self.transform:

            augmented = self.transform(
                image=image,
                mask=mask
            )

            image = augmented["image"]

            mask = augmented["mask"]

        # =========================
        # CONVERT TO TENSORS
        # =========================

        image = torch.tensor(
            image,
            dtype=torch.float32
        ).permute(2, 0, 1)

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        ).unsqueeze(0)

        return image, mask