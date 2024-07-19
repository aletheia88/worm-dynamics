from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Tuple, Dict
from wormdynamics.dataset import WormDataset
from wormdynamics.log import Log
from wormdynamics.model import WormTransformer, Embeddings
from wormdynamics.model_unet import UNet
from wormdynamics.parameters import (
        TransformerParameters, UNetParameters, DataParameters)
import glob
import itertools
import json
import numpy as np
import random
import torch


class GridSearchUNetNoiseMultiplier():

    def __init__(
            self,
            device: str,
            train_dataset_paths: List[str],
            valid_dataset_paths: List[str]):

        self.device = device
        self.train_dataset_paths = train_dataset_paths
        self.valid_dataset_paths = valid_dataset_paths
        self.model_parameters = UNetParameters(
                learning_rate=3e-4,
                max_epochs=500,
                eval_epochs=1,
                batch_size=1,
                device=self.device)
        self.log = Log(None)

    def generate_datasets(
            self,
            param_grid: Dict) -> List[Tuple[WormDataset, WormDataset]]:

        datasets = {}
        for combination in itertools.product(*param_grid.values()):
            config = "_".join(map(str, list(combination)))
            train_dataset = self._declare_dataset(
                    self.train_dataset_paths,
                    *combination)
            valid_dataset = self._declare_dataset(
                    self.valid_dataset_paths,
                    *combination)
            datasets[config] = [train_dataset, valid_dataset]

        return datasets

    def fit_models(self, param_grid: Dict):

        datasets = self.generate_datasets(param_grid)
        model_losses = {}

        for config, (train_dataset, valid_dataset) in datasets.items():

            model = self._declare_model()

            # each train-valid pair contains different data parameters
            model_losses[config] = {"train": {}, "valid": {}}
            self._fit_model(model, train_dataset, valid_dataset)
            model_losses[config]["train"] = [log.train_loss for log in self.log.iteration_logs]
            model_losses[config]["valid"] = [log.valid_loss for log in self.log.iteration_logs]
            with open("grid_search/outcomes.json", "w") as f:
                json.dump(model_losses, f, indent=4)
            # empty previous logs
            self.log.iteration_logs = []

    def _fit_model(self, model, train_dataset, valid_dataset):

        train_dataloader = self._declare_dataloader(train_dataset,
                                                    shuffle=True)
        valid_dataloader = self._declare_dataloader(valid_dataset,
                                                    shuffle=False)
        optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=self.model_parameters.learning_rate)

        for n_epoch in tqdm(range(self.model_parameters.max_epochs)):

            self.log.log_iteration(n_epoch, train_loss=0.0)

            for i, (inputs, targets, label) in enumerate(train_dataloader):

                augmented_inputs = train_dataset.augment_with_gaussian_noise(
                        inputs, label[0])
                mask_index = random.choice([0, 1])
                augmented_inputs[0, :, mask_index] = 0
                inputs = inputs.transpose(2, 1)
                targets = targets.transpose(2, 1)

                outputs, loss = model(inputs.float(), targets.float())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                self.log.iteration_logs[-1].train_loss += loss.item()

            self.log.iteration_logs[-1].train_loss /= len(train_dataloader.dataset)

            if n_epoch % self.model_parameters.eval_epochs == 0:
                #self.log.log_iteration(n_epoch, valid_loss=0.0)
                self.log.iteration_logs[-1].valid_loss = 0.0
                self.get_validation_loss(model, valid_dataloader)

    def _declare_dataset(self, dataset_paths, *combination):

        return WormDataset(DataParameters(
            dataset_paths,
            *combination,
            device=self.device))

    def _declare_dataloader(self, dataset, shuffle):

        return DataLoader(
                dataset,
                batch_size=self.model_parameters.batch_size,
                shuffle=shuffle)

    def _declare_model(self,):

        model = UNet().to(self.model_parameters.device)
        for param in model.parameters():
            param.data = param.data.float()

        return model

    @torch.no_grad()
    def get_validation_loss(self, model, valid_dataloader):

        model.eval() 
        for i, (inputs, targets, label) in enumerate(valid_dataloader):

            mask_index = random.choice([0, 1])
            inputs[0, :, mask_index] = 0
            inputs = inputs.transpose(2, 1)
            targets = targets.transpose(2, 1)
            outputs, loss = model(inputs, targets)

            self.log.iteration_logs[-1].valid_loss += loss.item()

        self.log.iteration_logs[-1].valid_loss /= len(valid_dataloader.dataset)
        model.train()


class GridSearchTransformer():

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
                *combination,
                device=self.device)
        # disable logging attention scores
        log = Log(None)
        model = WormTransformer(model_parameters,
                                log).to(model_parameters.device)
        for param in model.parameters():
            param.data = param.data.double()

        return model


def test_grid_search_transformer():
    dataset_paths = \
    glob.glob(f"/storage/fs/store1/alicia/transformer/AVA/*.json")
    train_paths = dataset_paths[:-2]
    valid_paths = dataset_paths[-2:]
    model_param_grid = {
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
    grid_search = GridSearchTransformer(device="cuda:1")
    models = grid_search.genernate_models(small_param_grid)
    grid_search.fit_models(small_param_grid, train_paths, valid_paths)


def test_grid_search_noise_multiplier():
    dataset_paths = \
    glob.glob(f"/storage/fs/store1/alicia/transformer/AVA/*.json")
    train_paths = dataset_paths[:-10]
    valid_paths = dataset_paths[-10:]
    noise = np.linspace(0, 1, 25)
    noise_multipliers = np.array([np.float32(np.round(num, 2)) for num in noise], dtype=np.float32)
    print(noise_multipliers)
    data_param_grid = {
            "neurons": [["AVA"]],
            "behaviors": [["velocity"]],
            "noise_multiplier": noise_multipliers,
            "num_to_augment": [0],
            "take_all": [True],
            "ignore_LRDV": [True],
    }
    device = "cuda:3"
    grid_search = GridSearchUNetNoiseMultiplier(device, train_paths,
                                                valid_paths)
    grid_search.fit_models(data_param_grid)


if __name__ == "__main__":
    test_grid_search_noise_multiplier()

