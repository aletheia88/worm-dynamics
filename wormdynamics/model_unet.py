""" This module incorporates code from Aladdin Persson
Original source:
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/model.py
Adapted by: Alicia Lu
Date: 2024-07-04 """
from torch.utils.data import DataLoader
from tqdm import tqdm
from wormdynamics.dataset import WormDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class DoubleConv2D(nn.Module):
    """ 2-D convolution applied twice"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet2D(nn.Module):
    def __init__(
            self, in_channels=4, out_channels=4,
            #features=[128, 256, 512, 1024]
            features=[64, 128, 256, 512],
    ):
        super(UNet2D, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Down part of U-Net
        for feature in features:
            self.downs.append(DoubleConv2D(in_channels, feature))
            in_channels = feature

        # Up part of U-Net
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv2D(feature*2, feature))

        self.bottleneck = DoubleConv2D(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x, targets):
        skip_connections = []
        #print(x.shape)
        for down in self.downs:
            x = down(x)
            #print(f"down: {x.shape}")
            skip_connections.append(x)
            x = x.squeeze(-1)
            x = self.pool(x)
            #print(f"pool: {x.shape}")
            x = x.unsqueeze(-1)
            #print(f"x shape: {x.shape}")

        x = self.bottleneck(x)
        #print(f"bottleneck: {x.shape}")
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            #print(f"up: {x.shape}")
            skip_connection = skip_connections[idx//2]
            #print(f"skip_connection: {skip_connection.shape}")
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            #print(f"concat_skip: {concat_skip.shape}")
            x = self.ups[idx+1](concat_skip)
            #print(f"up: {x.shape}")

        pred = self.final_conv(x)
        #print(f"final: {pred.shape}")
        if targets is None:
            loss = None
        else:
            loss = F.mse_loss(pred, targets)

        return pred, loss


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(
            self, in_channels=2, out_channels=2, features=[64, 128, 256, 512],
    ):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose1d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)

    def forward(self, x, targets):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            #print(f"down: {x.shape}")
            skip_connections.append(x)
            x = self.pool(x)
            #print(f"pool: {x.shape}")

        x = self.bottleneck(x)
        #print(f"bottleneck: {x.shape}")
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            #print(f"up: {x.shape}")
            skip_connection = skip_connections[idx//2]
            #print(f"skip_connection: {skip_connection.shape}")
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            #print(f"concat_skip: {concat_skip.shape}")
            x = self.ups[idx+1](concat_skip)
            #print(f"up: {x.shape}")

        pred = self.final_conv(x)
        #print(f"final: {pred.shape}")
        if targets is None:
            loss = None
        else:
            loss = F.mse_loss(pred, targets)

        return pred, loss


def testUNet1D_toy():
    num_dim = 4
    x = torch.randn((1, num_dim, 1600))
    y = torch.randn((1, num_dim, 1600))
    model = UNet(in_channels=num_dim, out_channels=num_dim)
    preds, loss = model(x, y)
    print(f"final shape: {preds.shape}")
    assert preds.shape == x.shape

def testUNet2D_toy():
    num_dim = 4
    # inputs: (batch_size, num_features, height, width)
    x = torch.randn((1, num_dim, 1600, 1))
    y = torch.randn((1, num_dim, 1600, 1))
    model = UNet2D(in_channels=num_dim, out_channels=num_dim)
    preds, loss = model(x, y)
    print(f"final shape: {preds.shape}")
    assert preds.shape == x.shape

def testUNet():
    train_files = [
        "2023-03-07-01_AVA.csv",
        "2022-07-20-01_AVA.csv",
        "2023-01-19-22_AVA.csv",
        "2023-01-23-15_AVA.csv",
    ]
    train_paths = [f"/home/alicia/store1/alicia/transformer/{file}" for file in
                   train_files]
    device = "cuda:3"
    train_set = WormDataset(train_paths, device, shift=0)
    train_dataloader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=True)
    model = UNet().to(device)
    for param in model.parameters():
        param.data = param.data.float()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    for epoch in tqdm(range(num_epochs)):
        for inputs, targets in train_dataloader:
            optimizer.zero_grad()
            inputs = inputs.transpose(2, 1)
            targets = targets.transpose(2, 1)
            outputs, loss = model(inputs, targets)
            loss.backward()
            optimizer.step()
    print(f"final loss: {loss.item()}")

if __name__ == "__main__":
    testUNet2D_toy()

