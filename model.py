from log import Log
from torch.nn import functional as F
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn

batch_size = 4
head_size = 2
block_size = 1600

n_head = 1
n_embd = 2
n_layer = 2
dropout = 0.2

learning_rate = 3e-4
device = "cuda:2"
torch.manual_seed(42)

def batch_data(file_names):

    data_list = []

    for idx, file_name in enumerate(file_names):
        df = pd.read_csv(f"data/{file_name}")
        AVAR = df.iloc[0].values[1:].astype(np.float64)
        AVAL = np.array(df.columns.values[1:], dtype=np.float64)

        combined_data = np.array([AVAL, AVAR]).T

        data_list.append(combined_data)

    data = np.stack(data_list, axis=0)
    return data


def initialize(inputs, attention_log_freq, loss_log_freq):

    B, T, C = inputs.shape
    idx = random.choice(list(range(B)))

    positional_embeddings = torch.from_numpy(
        get_positional_encoding(T, C)).to(device)

    logger = Log(attention_log_freq, loss_log_freq)
    model = WormTransformer(positional_embeddings, logger).to(device)

    for param in model.parameters():
        param.data = param.data.double()

    return model, logger

def get_positional_encoding(max_seq_length, d_model):

    position_embedding = np.zeros((max_seq_length, d_model))

    for pos in range(max_seq_length):
        for i in range(0, d_model, 2):
            position_embedding[pos, i] = \
                    np.sin(pos / (10000 ** ((2 * i) / d_model)))
            if i + 1 < d_model:
                position_embedding[pos, i + 1] = \
                    np.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

    return position_embedding


def get_batch(inputs):

    B, T, C = inputs.shape
    idx = random.choice(list(range(B)))
    target_embeddings = inputs[idx, :, :].double().view(1, T, C).to(device)
    input_embeddings = inputs[idx, :, :].double().view(1, T, C).to(device)

    return input_embeddings, target_embeddings


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size, logger):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril',
                torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.logger = logger

    def forward(self, x):
        # input of size (batch, time-step, embedding size)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)   # (B, T, hs)
        q = self.query(x) # (B, T, hs)
        # compute attention scores ("affinities")
        # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        # masking out future timepoints  # (B, T, T)
        #wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        self.logger.log_attention(wei)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size, logger):
        super().__init__()
        self.heads = nn.ModuleList(
                [Head(head_size, logger) for _ in range(num_heads)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(out)
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head, logger):
        # n_embd: embedding dimension
        # n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, logger)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # residual blocks (aka. skip connections)
        x = x + self.sa(x.double())
        x = x + self.ffwd(self.ln2(x.double()))
        return x


class WormTransformer(nn.Module):

    def __init__(self, positional_embeddings, logger):

        super().__init__()
        self.logger = logger

        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.position_embedding_table.weight = nn.Parameter(positional_embeddings)
        self.position_embedding_table.weight.requires_grad = False

        self.blocks = nn.Sequential(
                *[Block(n_embd, n_head=n_head, logger=logger)
                    for _ in range(n_layer)])
        # final layer norm: not used in our case
        self.ln_f = nn.LayerNorm(n_embd)
        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_embeddings, target_embeddings):

        B, T, C = input_embeddings.shape 
        # idx and targets are both (B,T) tensor of integers
        # we take token embedding to be whole-brain activities at a particular time point
        mask = torch.ones_like(input_embeddings[0, :, :])
        mask[:, 1] = 0
        input_emb = input_embeddings * mask
        pos_emb = self.position_embedding_table(
                torch.arange(T, device=device)).clone() * mask
        x = input_emb + pos_emb
        y = self.blocks(x.view(B, T, C))

        if target_embeddings is None:
            loss = None
        else:
            B, T, C = y.shape
            y = y.view(B*T, C)
            target_embeddings = target_embeddings.view(B*T, C)
            loss = F.mse_loss(y, target_embeddings)
            #mse_col0 = F.mse_loss(y[:, 0], target_embeddings[:, 0])
            #mse_col1 = F.mse_loss(y[:, 1], target_embeddings[:, 1])
            #loss = mse_col0 + 10 * mse_col1 
            #loss = F.mse_loss(y[:, 1], target_embeddings[:, 1])

        return y, loss


def main():

    file_names = [
        "2023-03-07-01_AVA.csv",
        "2022-07-20-01_AVA.csv",
        "2023-01-19-22_AVA.csv",
        "2023-01-23-15_AVA.csv",
    ]
    data = batch_data(file_names)
    max_iters = 5000
    tensor_data = torch.tensor(data)
    attention_log_freq, loss_log_freq = 100, 1

    model, logger = initialize(tensor_data, attention_log_freq, loss_log_freq)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for num_iter in tqdm(range(max_iters)[:]):

        logger.log_iteration(num_iter)
        xb, yb = get_batch(tensor_data)
        y, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logger.log_loss(loss.item())

    return model, logger
