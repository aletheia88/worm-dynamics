from torch.utils.data import DataLoader
from tqdm import tqdm
from wormdynamics.dataset import WormDataset
from wormdynamics.log import Log
from wormdynamics.model import WormTransformer, Embeddings
from wormdynamics.parameters import Parameters as ModelParameters
import random
import torch


def get_validation_loss(
        model: WormTransformer,
        valid_dataloader: DataLoader,
        embed: Embeddings):

    model.eval()
    model.log.iteration_logs[-1].valid_loss = 0.0

    with torch.no_grad():

        for input_embeddings, target_embeddings in valid_dataloader:

            embed.update_mask_embeddings(random.choice([0, 1]))
            positional_embeddings, mask_embeddings = embed.get_embeddings()

            mask_index = random.choice([0, 1])
            input_embeddings[0, :, mask_index] = 0

            inputs = input_embeddings + positional_embeddings + mask_embeddings
            y, valid_loss = model(inputs, target_embeddings)
            model.log.iteration_logs[-1].valid_loss += valid_loss.item()

    model.log.iteration_logs[-1].valid_loss /= \
            len(valid_dataloader.dataset)
    model.train()

train_files = [
    "2023-03-07-01_AVA.csv",
    "2022-07-20-01_AVA.csv",
    "2023-01-19-22_AVA.csv",
    "2023-01-23-15_AVA.csv",
]
train_paths = [f"/home/alicia/store1/alicia/transformer/{file}" for file in train_files]

valid_files = [
    "2023-01-19-01_AVA.csv",
    "2022-06-14-13_AVA.csv",
    "2022-08-02-01_AVA.csv",
    "2022-06-28-07_AVA.csv",
]
valid_paths = [f"/home/alicia/store1/alicia/transformer/{file}" for file in valid_files]

time_shift = 50 # shift AVAR

model_parameters = ModelParameters(
        n_layer=1,
        dropout=0.1,
        learning_rate=3e-4,
        max_epochs=5000,
        eval_epochs=10,
        batch_size=1,
        head_size=2,
        block_size=1600-time_shift,
        n_embd=2,
        ffwd_dim=4,
        device="cuda:1")

train_set = WormDataset(train_paths, model_parameters.device, shift=time_shift)
train_dataloader = DataLoader(
    train_set,
    batch_size=model_parameters.batch_size,
    shuffle=True)

valid_set = WormDataset(valid_paths, model_parameters.device, shift=time_shift)
valid_dataloader = DataLoader(
    valid_set,
    batch_size=model_parameters.batch_size,
    shuffle=False)

model = WormTransformer(
    model_parameters,
    Log(attention_log_freq=None)).to(model_parameters.device)

for param in model.parameters():
    param.data = param.data.double()
    
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=model_parameters.learning_rate)

embed = Embeddings(model_parameters, mask_index=0)

for n_epoch in tqdm(range(model_parameters.max_epochs)):

    model.log.log_iteration(n_epoch, train_loss=0.0)

    for i, (input_embeddings, target_embeddings) in enumerate(train_dataloader):

        embed.update_mask_embeddings(random.choice([0, 1]))
        positional_embeddings, mask_embeddings = embed.get_embeddings()

        mask_index = random.choice([0, 1])
        input_embeddings[0, :, mask_index] = 0

        inputs = input_embeddings + positional_embeddings + mask_embeddings

        y, loss = model(inputs, target_embeddings)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        model.log.iteration_logs[-1].train_loss += loss.item()

    model.log.iteration_logs[-1].train_loss /= \
                        len(train_dataloader.dataset)

    if n_epoch % model_parameters.eval_epochs == 0:
        get_validation_loss(model, valid_dataloader, embed)

