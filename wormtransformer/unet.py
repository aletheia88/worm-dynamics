from torch.utils.data import DataLoader
from tqdm import tqdm
from wormtransformer.dataset import WormDataset
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet1D(nn.Module):

    def __init__(self):
        super(UNet1D, self).__init__()

        ### encoder
        self.enc_conv1 = nn.Conv1d(2, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2) # reduce size to 800
        self.enc_conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2) # reduce size to 400

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
        x = F.relu(self.enc_conv1(x))
        print(f"enc_conv1: {x.shape}")
        x = self.pool1(x)
        print(f"pool1: {x.shape}")
        x = F.relu(self.enc_conv2(x))
        print(f"enc_conv2: {x.shape}")
        x = self.pool2(x)
        print(f"pool2: {x.shape}")

        ### bottleneck
        x = F.relu(self.bottleneck_conv(x))
        print(f"bottleneck: {x.shape}")
        ### decoder
        x = self.upconv1(x)
        print(f"upconv1: {x.shape}")
        x = F.relu(self.dec_conv1(x))
        print(f"dec_conv1: {x.shape}")
        x = self.upconv2(x)
        print(f"upconv2: {x.shape}")
        x = F.relu(self.dec_conv2(x))
        print(f"dec_conv2: {x.shape}")
        # final layer
        x = self.final_conv(x)
        print(f"final_conv: {x.shape}")
        return x

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
model = UNet1D().to(device)
for param in model.parameters():
    param.data = param.data.float()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5
for epoch in range(num_epochs):
    for inputs, targets in train_dataloader:
        optimizer.zero_grad()
        print(inputs.transpose(2, 1).shape)
        print(targets.shape)
        outputs = model(inputs.transpose(2, 1).float())
        loss = criterion(outputs, targets.transpose(2, 1))
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

