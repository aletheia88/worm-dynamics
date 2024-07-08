from torch.utils.data import DataLoader, Dataset
from wormtransformer.parameters import Parameters
import glob
import json
import numpy as np
import pandas as pd
import random
import torch


class WormDataset(Dataset):

    def __init__(self, dataset_paths, device, shift=0):

        self.shift = shift
        self.neuron_id_per_dataset = self._map_neurons(dataset_paths)
        self.dataset_paths = self._filter_datasets(dataset_paths, take_all=True)
        self.input_embeddings = torch.tensor(
                self.assemble_shuffled_neural_behavior_data(False, False, True), device=device)
        self.target_embeddings = self.input_embeddings.clone()

    def _filter_datasets(self, dataset_paths, take_all):

        if take_all:
            return dataset_paths

        filtered_dataset_paths = []
        for dataset_path in dataset_paths:
            dataset_name = dataset_path.split("/")[-1].split('.')[0]
            if len(self.neuron_id_per_dataset[dataset_name]) == 2:
                filtered_dataset_paths.append(dataset_path)

        return filtered_dataset_paths

    def _map_neurons(self, dataset_paths):

        neuron_id_per_dataset = dict()
        for dataset_path in dataset_paths:
            dataset_name = dataset_path.split("/")[-1].split('.')[0]
            neuron_id_per_dataset[dataset_name] = {}
            with open(dataset_path, "r") as f:
                data = json.load(f)

            for n_id, info_dict in data["labeled"].items():
                if info_dict['label'] == "AVAR" or info_dict['label'] == "AVAL":
                    neuron_id_per_dataset[dataset_name][info_dict['label']] = \
                            int(n_id) - 1
        return neuron_id_per_dataset

    def assemble_data(self,):
        """ neural activities of AVAL and AVAR """
        assembled_dataset = []
        for dataset_path in self.dataset_paths:

            dataset_name = dataset_path.split("/")[-1].split('.')[0]
            with open(dataset_path, "r") as f:
                trace = np.array(json.load(f)["trace_array"], dtype=np.float32).T
            AVAL_id = self.neuron_id_per_dataset[dataset_name]["AVAL"]
            AVAR_id = self.neuron_id_per_dataset[dataset_name]["AVAR"]
            AVAL = self._normalize(trace[:1210, AVAL_id])
            AVAR = self._normalize(trace[:1210, AVAR_id])
            AVAR_shifted = np.roll(AVAR, shift=self.shift)
            assembled_dataset.append(np.array([AVAL[self.shift:],
                                               AVAR_shifted[self.shift:]]).T)

        return np.stack(assembled_dataset, axis=0)

    def assemble_neural_behavior_data(self,):
        """ neural activities of AVAL/AVAR and velocity """

        assembled_dataset = []
        for dataset_path in self.dataset_paths:

            dataset_name = dataset_path.split("/")[-1].split('.')[0]
            with open(dataset_path, "r") as f:
                data = json.load(f)
                trace = np.array(data["trace_array"], dtype=np.float32).T
                behavior = np.array(data["velocity"], dtype=np.float32)
            id_dict = self.neuron_id_per_dataset[dataset_name]
            if len(id_dict) == 2:
                AVA_id = random.choice(list(id_dict.values()))
            elif len(id_dict) == 1:
                AVA_id = list(id_dict.values())[0]
            AVA = self._normalize(trace[:1600, AVA_id])
            velocity = self._normalize(behavior[:1600])
            assembled_dataset.append(np.array([AVA, velocity]).T)

        return np.stack(assembled_dataset, axis=0)

    def assemble_shuffled_neural_behavior_data(
            self,
            shuffle_animal,
            shuffle_trace,
            shuffle_behavior):
        """ create datasets with shuffled neural activities or behaviors """

        assembled_dataset = []
        for i, dataset_path in enumerate(self.dataset_paths):

            dataset_name = dataset_path.split("/")[-1].split('.')[0]
            id_dict = self.neuron_id_per_dataset[dataset_name]
            if len(id_dict) == 2:
                AVA_id = random.choice(list(id_dict.values()))
            elif len(id_dict) == 1:
                AVA_id = list(id_dict.values())[0]

            if not shuffle_animal:
                with open(dataset_path, "r") as f:
                    data = json.load(f)
                    trace = np.array(data["trace_array"], dtype=np.float32).T
                    behavior = np.array(data["velocity"], dtype=np.float32)

                AVA = trace[:1600, AVA_id]

                if shuffle_trace:
                    random.shuffle(AVA)
                elif shuffle_behavior:
                    random.shuffle(behavior)
                else:
                    raise NotImplementedError(
                    "either shuffle trace or shuffle behavior")
            else:
                if not shuffle_trace and not shuffle_behavior:
                    if i == len(self.dataset_paths) - 1:
                        j = 0
                    else:
                        j = i + 1
                    dataset_path2 = self.dataset_paths[j]
                    with open(dataset_path, "r") as f:
                        data = json.load(f)
                        trace = np.array(data["trace_array"],
                                         dtype=np.float32).T
                    AVA = trace[:1600, AVA_id]

                    with open(dataset_path2, "r") as f:
                        data = json.load(f)
                        behavior = np.array(data["velocity"], dtype=np.float32)
                else:
                    raise NotImplementedError(
                    "no need to shuffle trace or behavior")

            # normalize neural and behavior activities
            AVA = self._normalize(AVA)
            velocity = self._normalize(behavior[:1600])
            assembled_dataset.append(np.array([AVA, velocity]).T)

        return np.stack(assembled_dataset, axis=0)

    def _normalize(self, data):
        return (data - data.min()) / (data.max() - data.min()) * 2 - 1

    def __getitem__(self, index):
        x = self.input_embeddings[index]
        y = self.target_embeddings[index]
        return x, y

    def __len__(self):
        return len(self.input_embeddings)

def test():
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
    #dataset_paths = [f"/home/alicia/store1/alicia/transformer/{file}" for file in train_files]
    dataset_paths = glob.glob(f"/storage/fs/store1/alicia/transformer/AVA/*.json")
    parameters = Parameters(
            n_layer=1,
            dropout=0.1,
            learning_rate=3e-4,
            max_epochs=10,
            eval_epochs=10,
            batch_size=1,
            head_size=2,
            block_size=1600,
            n_embd=2,
            ffwd_dim=4,
            attention_span=10,
            device="cuda:3")
    dataset = WormDataset(dataset_paths, parameters.device)
    dataloader = DataLoader(dataset, batch_size=parameters.batch_size,
                            shuffle=True)
    print(len(dataloader.dataset))
    for i, (inputs, targets) in enumerate(dataloader):
        print(f"batch {i}, inputs: {inputs.shape} targets: {targets.shape}\n")

if __name__ == "__main__":
    test()
