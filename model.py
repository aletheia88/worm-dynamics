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


def initialize(inputs, attention_log_freq):

    B, T, C = inputs.shape
    idx = random.choice(list(range(B)))

    positional_embeddings = torch.from_numpy(
        get_positional_encoding(T, C)).to(device)

    log = Log(attention_log_freq)
    model = WormTransformer(positional_embeddings, log).to(device)

    for param in model.parameters():
        param.data = param.data.double()

    return model, log


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

    def normalize(embeddings):

        normalized = np.zeros_like(embeddings, dtype=np.float32)
        for idx in range(embeddings.shape[1]):
            min_val = min(embeddings[:, idx])
            max_val = max(embeddings[:, idx])
            if min_val == max_val:
                normalized[:, idx] = 0
            else:
                normalized[:, idx] = (embeddings[:, idx] - min_val) / (max_val
                        - min_val) * 2 - 1

        return normalized

    B, T, C = inputs.shape
    idx = random.choice(list(range(B)))
    normalized_inputs = normalize(inputs[idx, :, :])
    # negate the AVAL neuron
    normalized_inputs[:, 0] = - normalized_inputs[:, 0]
    target_embeddings = inputs[idx, :, :].double().view(1, T, C).to(device)
    input_embeddings = inputs[idx, :, :].double().view(1, T, C).to(device)

    return input_embeddings, target_embeddings


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size, log):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril',
                torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.log = log

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
        self.log.log_attention(wei)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size, log):
        super().__init__()
        self.heads = nn.ModuleList(
                [Head(head_size, log) for _ in range(num_heads)])
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

    def __init__(self, n_embd, n_head, log):
        # n_embd: embedding dimension
        # n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, log)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # residual blocks (aka. skip connections)
        x = x + self.sa(x.double())
        x = x + self.ffwd(self.ln2(x.double()))
        return x


class WormTransformer(nn.Module):

    def __init__(self, positional_embeddings, log):

        super().__init__()
        self.log = log

        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.position_embedding_table.weight = nn.Parameter(positional_embeddings)
        self.position_embedding_table.weight.requires_grad = False

        self.blocks = nn.Sequential(
                *[Block(n_embd, n_head=n_head, log=log)
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

    def forward(self, input_embeddings, target_embeddings, mask_index):

        B, T, C = input_embeddings.shape
        # idx and targets are both (B,T) tensor of integers
        # we take token embedding to be whole-brain activities at a particular time point
        mask = torch.ones((T, C)).to(device)
        mask[:, mask_index] = 0

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

        return y, loss

@torch.no_grad()
def estimate_loss():

    model.eval()
    xb, yb = get_batch(valid_data)
    y, valid_loss = model(xb, yb, mask_index=1)
    log.iteration_logs[-1].valid_loss = valid_loss.item()

    model.train()


train_files = [
    "2023-03-07-01_AVA.csv",
    "2022-07-20-01_AVA.csv",
    "2023-01-19-22_AVA.csv",
    "2023-01-23-15_AVA.csv",
]
valid_files = [
    "2023-01-19-01_AVA.csv",
    "2022-06-14-13_AVA.csv",
    "2022-08-02-01_AVA.csv",
    "2022-06-28-07_AVA.csv",
]
test_files = [
    "2022-07-15-06_AVA.csv",
    "2022-07-15-12_AVA.csv",
]

train_data = batch_data(train_files)
valid_data = batch_data(valid_files)
test_data = batch_data(test_files)
print(train_data.shape, valid_data.shape, test_data.shape)

batch_size = 4
max_iters = 5000
eval_iters = 20

head_size = 2
block_size = 1600

n_head = 1
n_embd = 2
n_layer = 3
dropout = 0.5

learning_rate = 3e-4
device = "cuda:3"

attention_log_freq, loss_log_freq = 100, 1

model, log = initialize(torch.tensor(train_data), attention_log_freq)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for num_iter in tqdm(range(max_iters)[:]):

    log.log_iteration(num_iter)
    xb, yb = get_batch(train_data)
    y, loss = model(xb, yb, mask_index=1)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    log.iteration_logs[-1].train_loss = loss.item()
    if num_iter % eval_iters == 0:
        estimate_loss()

