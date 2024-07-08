from torch.utils.data import DataLoader
from tqdm import tqdm
from wormtransformer.dataset import WormDataset
from wormtransformer.log import Log
from wormtransformer.model_unet import UNET
import glob
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


dataset_paths = glob.glob(f"/storage/fs/store1/alicia/transformer/AVA/*.json")
train_paths = dataset_paths[:-2]
valid_paths = dataset_paths[-2:]

device = "cuda:3"
learning_rate = 3e-4
max_epochs = 1000
eval_epochs = 1
batch_size = 1
shift = 10

train_set = WormDataset(train_paths, device, shift=shift)
train_dataloader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True)

valid_set = WormDataset(valid_paths, device, shift=shift)
valid_dataloader = DataLoader(
    valid_set,
    batch_size=batch_size,
    shuffle=False)

model = UNET().to(device)

for param in model.parameters():
    param.data = param.data.float()

optimizer = torch.optim.AdamW(model.parameters(),
                              lr=learning_rate)
log = Log(attention_log_freq=None)

@torch.no_grad()
def get_validation_loss(model, valid_dataloader):

    model.eval()

    for i, (inputs, targets) in enumerate(valid_dataloader):

        mask_index = random.choice([0, 1])
        inputs[0, :, mask_index] = 0

        optimizer.zero_grad()
        inputs = inputs.transpose(2, 1)
        targets = targets.transpose(2, 1)
        outputs, loss = model(inputs, targets)

        log.iteration_logs[-1].valid_loss += loss.item()

    log.iteration_logs[-1].valid_loss /= len(valid_dataloader.dataset)
    model.train()

    return inputs, outputs, targets, mask_index

for n_epoch in tqdm(range(max_epochs)):

    log.log_iteration(n_epoch, train_loss=0.0)

    for i, (inputs, targets) in enumerate(train_dataloader):

        mask_index = random.choice([0, 1])
        inputs[0, :, mask_index] = 0
        inputs = inputs.transpose(2, 1)
        targets = targets.transpose(2, 1)
        outputs, loss = model(inputs.float(), targets.float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        log.iteration_logs[-1].train_loss += loss.item()

    log.iteration_logs[-1].train_loss /= len(train_dataloader.dataset)

    if n_epoch % eval_epochs == 0:
        log.iteration_logs[-1].valid_loss = 0.0
        inputs_valid, outputs_valid, targets_valid = get_validation_loss(model, valid_dataloader)

