import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        # sub module 4
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=0,
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=0,
        )

        self.linear = nn.Linear(
            in_features=1568,
            out_features=125,
        )
        self.out = nn.Linear(
            in_features=125,
            out_features=2,
        )

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = h.reshape(h.shape[0], -1)
        h = F.relu(self.linear(h))
        y = self.out(h)
        return y
        # # input x pass conv1
        # # image shape 32,32,3 to batch size
        # # x shape = (batch_size,channels,height,width)
        # h = self.conv1(x)
        # # if x < 0 ? 0 by relu
        # h = F.relu(h)
        # h = F.relu(self.conv2(h))

        # # h shape = (batch_size,32 (features),h,w)

        # # reshape to remain 1 dim because linear need 1 dim
        # # h  -> (batch_size,32*h*w) to be long vector for linear
        # h = h.reshape(h.shape[0], -1)
        # h = F.relu(self.linear(h))
        # y_pred = self.out(h)
        # return y_pred
