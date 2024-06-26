from torch.utils.data import DataLoader, Dataset
from wormtransformer.parameters import Parameters
import numpy as np
import pandas as pd
import torch


class WormDataset(Dataset):

    def __init__(self, dataset_paths, device):

        self.input_embeddings = torch.tensor(
                self._assemble_data(dataset_paths), device=device)
        self.target_embeddings = self.input_embeddings.clone()

    def _assemble_data(self, dataset_paths):

        assembled_dataset = []
        for dataset_path in dataset_paths:
            df = pd.read_csv(dataset_path)
            AVAR = self._normalize(df.iloc[0].values[1:].astype(np.float64))
            AVAL = self._normalize(np.array(df.columns.values[1:],
                                            dtype=np.float64))
            assembled_dataset.append(np.array([AVAL, AVAR]).T)

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
    dataset_paths = [f"/home/alicia/store1/alicia/transformer/{file}" for file in train_files]
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
            device="cuda:3")
    dataset = WormDataset(dataset_paths, parameters.device)
    dataloader = DataLoader(dataset, batch_size=parameters.batch_size,
                            shuffle=True)
    print(len(dataloader.dataset))
    for i, (inputs, targets) in enumerate(dataloader):
        print(f"batch {i}, inputs: {inputs.shape} targets: {targets.shape}\n")
