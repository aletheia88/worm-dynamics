from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from wormtransformer.dataset import WormDataset
from wormtransformer.log import Log
from wormtransformer.parameters import Parameters as ModelParameters
import math
import numpy as np
import random
import torch
import torch.nn as nn


class Embeddings(nn.Module):

    def __init__(self, parameters):
        super().__init__()
        """
        positional_embeddings = torch.zeros(parameters.block_size,
                                            parameters.n_embd)
        position = torch.arange(0, parameters.block_size,
                                dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, parameters.n_embd, 2).float() *
                             (-math.log(10000.0) / parameters.n_embd))
        positional_embeddings[:, 0::2] = torch.sin(position * div_term)
        positional_embeddings[:, 1::2] = torch.cos(position * div_term)
        """
        positional_embeddings = nn.Parameter(self.encode_positions(parameters))
        self.register_buffer("positional_embeddings", positional_embeddings)
        mask_embeddings = torch.zeros(parameters.block_size, parameters.n_embd)
        self.register_buffer("mask_embeddings", mask_embeddings)

        self.device = parameters.device

    def encode_positions(self, parameters):
        positional_embeddings = np.zeros((parameters.block_size, parameters.n_embd))
        for pos in range(parameters.block_size):
            for i in range(0, parameters.n_embd, 2):
                positional_embeddings[pos, i] = np.sin(
                        pos / (10000 ** ((2 * i) / parameters.n_embd)))
                if i + 1 < parameters.n_embd:
                    positional_embeddings[pos, i + 1] = np.cos(
                            pos / (10000 ** ((2 * (i + 1)) / parameters.n_embd)))

        return torch.tensor(positional_embeddings)

    def update_mask_embeddings(self, mask_index):
        self.mask_embeddings.data.fill_(0.1)
        self.mask_embeddings.data[:, mask_index] = 0

    def get_embeddings(self):
        return self.positional_embeddings.to(self.device), \
                self.mask_embeddings.to(self.device)


class Head(nn.Module):

    """one head of self-attention"""

    def __init__(self, parameters: ModelParameters, log: Log):
        super().__init__()
        self.key = nn.Linear(parameters.n_embd,
                             parameters.head_size,
                             bias=False)
        self.query = nn.Linear(parameters.n_embd,
                               parameters.head_size,
                               bias=False)
        self.value = nn.Linear(parameters.n_embd,
                               parameters.head_size,
                               bias=False)
        self.register_buffer('tril',
                torch.tril(torch.ones(parameters.block_size,
                                      parameters.block_size)))
        self.dropout = nn.Dropout(parameters.dropout)
        self.log = log

    def forward(self, x):
        # input of size (batch_size, block_size, parameters.n_embd)
        # output of size (batch_size, block_size, head_size)
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        # masking out future timepoints  # (B, T, T)
        #wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        if self.log.attention_log_freq is not None:
            if len(self.log.iteration_logs) > 0 and \
                self.log.iteration_logs[-1].iteration % \
                self.log.attention_log_freq == 0:
                    self.log.iteration_logs[-1].attention = wei

        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):

    """multiple heads of self-attention in parallel"""

    def __init__(self, parameters: ModelParameters, log: Log):
        super().__init__()
        self.heads = nn.ModuleList(
                [Head(parameters, log) for _ in range(parameters.n_head)])
        self.dropout = nn.Dropout(parameters.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):

    """a simple linear layer followed by a non-linearity"""

    def __init__(self, parameters: ModelParameters):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(parameters.n_embd,
                      parameters.ffwd_dim * parameters.n_embd),
            nn.ReLU(),
            nn.Linear(parameters.ffwd_dim * parameters.n_embd,
                      parameters.n_embd),
            nn.Dropout(parameters.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    """transformer block: communication followed by computation"""

    def __init__(self, parameters: ModelParameters, log: Log):
        super().__init__()
        self.proj = nn.Linear(parameters.n_embd, parameters.n_embd)
        self.sa1 = MultiHeadAttention(parameters, log)
        #self.sa2 = MultiHeadAttention(parameters, log)
        self.ffwd = FeedForward(parameters)
        self.bn = nn.BatchNorm1d(parameters.block_size)
        #self.ln1 = nn.LayerNorm(parameters.n_embd)
        #self.ln2 = nn.LayerNorm(parameters.n_embd)

    def forward(self, x):
        # (1600, 2) x (2, 2) -> (1600, 2)
        x = self.proj(x.float())
        # (1600, 2) -> (1600, 2)
        x = x + self.sa1(x.float())
        #x = x + self.sa2(x.float())
        x = x + self.ffwd(self.bn(x.float()))
        return x


class MASELoss(nn.Module):

    """mean absolute scaled error"""

    def __init__(self):
        super(MASELoss, self).__init__()

    def forward(self, y_pred, y_true):

        y_naive = torch.roll(y_true, shifts=1, dims=1)
        y_naive[:, 0] = y_true[:, 0]

        mean_naive_errors = torch.abs(y_true - y_naive).mean()
        mean_prediction_errors = torch.abs(y_true - y_pred).mean()

        return mean_prediction_errors / mean_naive_errors


class WormTransformer(nn.Module):

    def __init__(self, model_parameters, log):

        super().__init__()
        self.log = log
        self.blocks = nn.Sequential(
                *[Block(model_parameters, log) for _ in range(model_parameters.n_layer)])
        # final layer norm: not used in our case
        self.ln_f = nn.LayerNorm(model_parameters.n_embd)
        self.apply(self._init_weights)
        self.model_parameters = model_parameters

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, inputs, target_embeddings):

        B, T, C = target_embeddings.shape
        y = self.blocks(inputs.view(B, T, C))

        if target_embeddings is None:
            loss = None
        else:
            B, T, C = y.shape
            y = y.view(B*T, C)
            target_embeddings = target_embeddings.view(B*T, C)
            mase_loss = MASELoss()
            loss = mase_loss(y, target_embeddings)
            #loss = F.mse_loss(y, target_embeddings)

        return y, loss


def test():
    train_files = [
        "2023-03-07-01_AVA.csv",
        "2022-07-20-01_AVA.csv",
        "2023-01-19-22_AVA.csv",
        "2023-01-23-15_AVA.csv",
    ]
    train_paths = [f"/home/alicia/store1/alicia/transformer/{file}" for file in
                   train_files]

    valid_files = [
        "2023-01-19-01_AVA.csv",
        "2022-06-14-13_AVA.csv",
        "2022-08-02-01_AVA.csv",
        "2022-06-28-07_AVA.csv",
    ]
    valid_paths = [f"/home/alicia/store1/alicia/transformer/{file}" for file in
                   valid_files]

    time_shift = 0 # shift AVAR

    model_parameters = ModelParameters(
            n_layer=1,
            dropout=0.1,
            learning_rate=3e-3,
            max_epochs=1000,
            eval_epochs=1,
            batch_size=1,
            head_size=2,
            block_size=1600-time_shift,
            n_embd=2,
            ffwd_dim=4,
            device="cuda:3")

    train_set = WormDataset(train_paths, model_parameters.device,
                            shift=time_shift)
    train_dataloader = DataLoader(
        train_set,
        batch_size=model_parameters.batch_size,
        shuffle=True)

    valid_set = WormDataset(valid_paths, model_parameters.device,
                            shift=time_shift)
    valid_dataloader = DataLoader(
        valid_set,
        batch_size=model_parameters.batch_size,
        shuffle=False)

    model = WormTransformer(
        model_parameters,
        Log(attention_log_freq=None)).to(model_parameters.device)

    for param in model.parameters():
        param.data = param.data.float()

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=model_parameters.learning_rate)

    embed = Embeddings(model_parameters)

    mask_index = random.choice([0, 1])

    for n_epoch in tqdm(range(model_parameters.max_epochs)):

        model.log.log_iteration(n_epoch, train_loss=0.0)

        for i, (input_embeddings, target_embeddings) in \
            enumerate(train_dataloader):

            embed.update_mask_embeddings(mask_index)
            positional_embeddings, mask_embeddings = embed.get_embeddings()
            input_embeddings[0, :, mask_index] = 0

            inputs = input_embeddings + positional_embeddings

            y, loss = model(inputs, target_embeddings)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            model.log.iteration_logs[-1].train_loss += loss.item()

        model.log.iteration_logs[-1].train_loss /= \
                            len(train_dataloader.dataset)

        # if n_epoch % model_parameters.eval_epochs == 0:
        #     get_validation_loss(model, valid_dataloader, embed)

if __name__ == "__main__":
    test()
