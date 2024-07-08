""" This module incorporates code from Aladdin Persson
Original source:
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/model.py
Adapted by: Alicia Lu
Date: 2024-07-04 """
from torch.utils.data import DataLoader
from tqdm import tqdm
from wormtransformer.dataset import WormDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


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

class UNET(nn.Module):
    def __init__(
            self, in_channels=2, out_channels=2, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
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
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        pred = self.final_conv(x)
        if targets is None:
            loss = None
        else:
            loss = F.mse_loss(pred, targets)

        return pred, loss

class UNet1D(nn.Module):
    """convolving over the feature dimension"""
    def __init__(self):
        super(UNet1D, self).__init__()

        ### encoder
        self.enc_conv1 = nn.Conv1d(2, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.enc_conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)

        ### bottleneck
        self.bottleneck_conv = nn.Conv1d(32, 64, kernel_size=3, padding=1)

        ### decoder
        self.upconv1 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)

        self.dec_conv1 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Conv1d(16, 16, kernel_size=3, padding=1)

        self.final_conv = nn.Conv1d(16, 2, kernel_size=3, padding=1)

    def forward(self, x):
        ### encoder
        # (1, 2, 1600) -> (1, 16, 1600)
        x = F.relu(self.enc_conv1(x))
        # (1, 16, 1600) -> (1, 16, 800)
        x = self.pool1(x)
        # (1, 16, 800) -> (1, 32, 800)
        x = F.relu(self.enc_conv2(x))
        # (1, 32, 800) -> (1, 32, 400)
        x = self.pool2(x)

        ### bottleneck
        # (1, 32, 400) -> (1, 64, 400)
        x = F.relu(self.bottleneck_conv(x))

        ### decoder
        # (1, 64, 400) -> (1, 32, 800)
        x = self.upconv1(x)
        # (1, 32, 800) -> (1, 32, 800)
        x = F.relu(self.dec_conv1(x))
        # (1, 32, 800) -> (1, 16, 1600)
        x = self.upconv2(x)
        # (1, 16, 1600) -> (1, 16, 1600)
        x = F.relu(self.dec_conv2(x))
        # final layer
        # (1, 16, 1600) -> (1, 2, 1600)
        x = self.final_conv(x)
        return x

def testUNET():
    x = torch.randn((1, 2, 1500))
    y = torch.randn((1, 2, 1500))
    model = UNET(in_channels=2, out_channels=2)
    preds, loss = model(x, y)
    print(preds.shape)
    assert preds.shape == x.shape

def test():
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
    model = UNET().to(device)
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
    testUNET()
