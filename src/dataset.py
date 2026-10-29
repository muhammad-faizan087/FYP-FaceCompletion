
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CelebAHQ256(Dataset):
    # fixed_masks=False -> new random mask on every load (training)
    # fixed_masks=True  -> reproducible mask for each image index (val/test)

    def __init__(
        self,
        img_dir,
        split_file,
        augment=False,
        mask_fn=None,
        fixed_masks=False,
        mask_seed=42
    ):
        self.img_dir = img_dir

        with open(split_file, "r") as f:
            self.names = f.read().split()

        self.augment = augment
        self.mask_fn = mask_fn
        self.fixed_masks = fixed_masks
        self.mask_seed = mask_seed

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        img = Image.open(
            os.path.join(self.img_dir, name)
        ).convert("RGB")

        img = np.asarray(img, dtype=np.float32) / 255.0

        if self.augment and np.random.rand() < 0.5:
            img = img[:, ::-1].copy()

        # [0, 1] -> [-1, 1]; shape: (3, H, W)
        image = torch.from_numpy(img).permute(2, 0, 1) * 2 - 1

        h, w = image.shape[1:]

        if self.mask_fn is None:
            raise ValueError("mask_fn must be provided.")

        if self.fixed_masks:
            rng = np.random.default_rng(self.mask_seed + idx)
            mask_np = self.mask_fn(h, w, rng=rng)
        else:
            mask_np = self.mask_fn(h, w)

        # Shape: (1, H, W); 1 = missing region
        mask = torch.from_numpy(mask_np).unsqueeze(0)

        masked = image * (1 - mask)

        return {
            "image": image,
            "masked": masked,
            "mask": mask,
            "name": name
        }

