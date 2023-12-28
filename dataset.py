import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# 1.create dataset object define that how to get images from path
class CatDogMiniDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        # Where is files
        self.target = pd.read_csv(image_dir + "annotations.csv")
        # transform or not
        self.transform = transform

    def __len__(self):
        # size
        return len(self.target)

    def __getitem__(self, index):
        # How to get data
        img_path = os.path.join(
            self.image_dir, self.target.iloc[index, 0]
        )  # row = index , column = 0 is file name in csv
        # get image then convert to numpy and get target to tensor int
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
        target_label = torch.tensor(int(self.target.iloc[index, 1]))

        # pre-process image to augment increase a lot of image
        if self.transform:
            image = self.transform(image=image)["image"]

        return (image, target_label)


if __name__ == "__main__":
    image_dir = "data/train/"
    # create dataset object
    dataset = CatDogMiniDataset(image_dir)
    # see some image
    img, target_label = dataset[0]
    print(img.shape)  # height,width,channel (RGB)

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    train_transform = A.Compose(
        [
            A.Resize(height=32, width=32),
            A.Normalize(  # -mean / std
                mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0
            ),
        ]  # Really crop and give ratio instead and  # Normalize input should be range 0-1 because weight is the same
    )

    dataset = CatDogMiniDataset(image_dir, train_transform)
    img, target_label = dataset[0]
    print(img.shape)
