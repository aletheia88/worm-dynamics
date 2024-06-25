from dataset import WormDataset
from log import Log
from model import WormTransformer, Embeddings
from parameters import Parameters as ModelParameters
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List
import itertools
import random
import torch


class GridSearch():

    def __init__(self, device):
        self.device = device

    def generate_models(self, param_grid: dict) -> List[dict]:
        models = {}
        for combination in itertools.product(*param_grid.values()):
            model = self._declare_model(combination)
            model_name = "_".join(map(str, list(combination)))
            models[model_name] = model

        return models

    def fit_models(
            self,
            param_grid: dict,
            train_paths: List[str],
            valid_paths: List[str]):

        train_set = WormDataset(train_paths, self.device)
        valid_set = WormDataset(valid_paths, self.device)

        models = self.generate_models(param_grid)

        best_results = [list(models.values())[0], float("inf"), "model_name"]
        for model_name, model in models.items():
            model_parameters = model.model_parameters
            train_dataloader = DataLoader(
                    train_set,
                    batch_size=model_parameters.batch_size,
                    shuffle=True)
            valid_dataloader = DataLoader(
                    valid_set,
                    batch_size=model_parameters.batch_size,
                    shuffle=False)
            optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=model_parameters.learning_rate)
            embed = Embeddings(model_parameters, mask_index=0)

            for n_epoch in tqdm(range(model_parameters.max_epochs)):

                model.log.log_iteration(n_epoch, train_loss=0)
                for n_iter, (input_embeddings,
                             target_embeddings) in enumerate(train_dataloader):

                    embed.update_mask_embeddings(random.choice([0, 1]))
                    positional_embeddings, mask_embeddings = embed.get_embeddings()
                    inputs = input_embeddings + positional_embeddings + mask_embeddings
                    y, loss = model(inputs, target_embeddings)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    model.log.iteration_logs[-1].train_loss += loss.item()

                model.log.iteration_logs[-1].train_loss /= \
                                    len(train_dataloader.dataset)

                if n_epoch % model_parameters.eval_epochs == 0:
                    self.get_validation_loss(model, valid_dataloader, embed)

                    # update the best model after each epoch
                    valid_loss = model.log.iteration_logs[-1].valid_loss
                    if valid_loss < best_results[1]:
                        best_results[0] = model
                        best_results[1] = valid_loss
                        best_results[2] = model_name

    def get_validation_loss(
            self,
            model: WormTransformer,
            valid_dataloader: DataLoader,
            embed: Embeddings):

        model.eval()
        model.log.iteration_logs[-1].valid_loss = 0.0

        with torch.no_grad():

            for input_embeddings, target_embeddings in valid_dataloader:

                embed.update_mask_embeddings(random.choice([0, 1]))
                positional_embeddings, mask_embeddings = embed.get_embeddings()
                inputs = input_embeddings + positional_embeddings + mask_embeddings
                y, valid_loss = model(inputs, target_embeddings)
                model.log.iteration_logs[-1].valid_loss += valid_loss.item()

        model.log.iteration_logs[-1].valid_loss /= \
                len(valid_dataloader.dataset)
        model.train()

    def _dump_memory(self, model):
        """ discard the items logged during model's training"""
        model.log.iteration_logs = []

    def _declare_model(self, combination):
        model_parameters = ModelParameters(
                n_layer=combination[0],
                dropout=combination[1],
                learning_rate=combination[2],
                max_epochs=combination[3],
                eval_epochs=combination[4],
                batch_size=combination[5],
                head_size=combination[6],
                block_size=combination[7],
                n_embd=combination[8],
                ffwd_dim=combination[9],
                device=self.device)
        # disable logging attention scores
        log = Log(None)
        model = WormTransformer(model_parameters,
                                log).to(model_parameters.device)
        for param in model.parameters():
            param.data = param.data.double()

        return model

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
    param_grid = {
        "n_layer": [1, 2, 3],
        "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
        "learning_rate": [3e-3, 3e-4],
        "max_epochs": 5000,
        "eval_epochs": 20,
        "batch_size": 1,
        "head_size": 2,
        "block_size": 1600,
        "n_embd": 2,
        "ffwd_dim": [2, 4, 6, 8],
    }
    small_param_grid = {
        "n_layer": [1, 2, 3],
        "dropout": [0.1],
        "learning_rate": [3e-3, 3e-4],
        "max_epochs": [10],
        "eval_epochs": [2],
        "batch_size": [1],
        "head_size": [2],
        "block_size": [1600],
        "n_embd": [2],
        "ffwd_dim": [2, 4],
    }

    grid_search = GridSearch(device="cuda:1")
    #models = grid_search.genernate_models(small_param_grid)
    grid_search.fit_models(small_param_grid, train_paths, valid_paths)

