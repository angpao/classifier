import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from dataset import CatDogMiniDataset
from logger import Logger
from model import MyNet


def main():
    # 1.define preprocessing
    train_transform = A.Compose(
        [
            A.Resize(height=32, width=32),
            A.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )
    # 2.create dataset object
    train_data_object = CatDogMiniDataset(
        image_dir="data/train/", transform=train_transform
    )

    # 3.create data loader
    train_loader = DataLoader(
        train_data_object,
        batch_size=32,
        num_workers=2,  # CPU core
        pin_memory=True,
        shuffle=True,  # load random images
    )
    # 4. Create Model object
    network = MyNet()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    network.to(device)

    # 5. Define loss func and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=0.001)

    # 6. Logger object
    logger = Logger(device)

    # 7. Training loop
    print("training is started")
    for epoch in range(1000):
        for batch_idx, (x, target) in enumerate(train_loader):
            # 8. set device
            x = x.to(device)
            target = target.to(device)
            # 9. make prediction
            y_pred = network(x)

            # 10. Compute loss
            loss = loss_fn(y_pred, target)

            # 11. Compute gradients
            optimizer.zero_grad()  # reset gradients
            loss.backward()  # compute gradient

            # 12 . adjust the weights
            optimizer.step()

            # 13. collect result into the logger
            logger.log_step(loss.item())


if __name__ == "__main__":
    main()
