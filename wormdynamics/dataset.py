from torch.utils.data import DataLoader, Dataset
from wormdynamics.parameters import UNetParameters, DataParameters
import copy
import glob
import json
import numpy as np
import pandas as pd
import random
import torch


class WormDataset(Dataset):

    def __init__(self, data_parameters):

        self.neurons = data_parameters.neurons
        self.behaviors = data_parameters.behaviors
        self.noise_multiplier = data_parameters.noise_multiplier
        self.device = data_parameters.device
        self.ignore_LRDV = data_parameters.ignore_LRDV
        self.take_all = data_parameters.take_all

        self.neuron_columns = {}    # column indices of actual neurons
        self.behavior_columns = {}  # column indices of animal behaviors

        self.neuron_id_per_dataset = self._map_neurons(
                data_parameters.dataset_paths)

        self.dataset_paths = self._filter_datasets(
                data_parameters.dataset_paths)

        self.labels = [path.split("/")[-1].split('.')[0] for path in
                       self.dataset_paths]

        self.input_embeddings = torch.tensor(
                self.assemble_neural_behavior_data(),
                    device=data_parameters.device)

        self.target_embeddings = self.input_embeddings.clone()

    def _filter_datasets(self, dataset_paths):
        """ Select datasets that contain all neurons of interest. """

        if self.take_all:
            return dataset_paths
        elif self.neurons:
            filtered_dataset_paths = []
            for dataset_path in dataset_paths:
                dataset_name = dataset_path.split("/")[-1].split('.')[0]

                if set(list(self.neuron_id_per_dataset[dataset_name].keys()
                        )) == set(self.neurons):
                    filtered_dataset_paths.append(dataset_path)
            return filtered_dataset_paths
        else:
            raise ValueError("Needs to indicate which neurons to take.")

    def _map_neurons(self, dataset_paths):

        neuron_id_per_dataset = dict()
        for dataset_path in dataset_paths:
            dataset_name = dataset_path.split("/")[-1].split('.')[0]
            neuron_id_per_dataset[dataset_name] = {}
            with open(dataset_path, "r") as f:
                data = json.load(f)

            for n_id, info_dict in data["labeled"].items():

                if not self.ignore_LRDV:
                    if info_dict['label'] in self.neurons:
                        neuron_name = info_dict['label']
                        neuron_id_per_dataset[dataset_name][neuron_name] = \
                                int(n_id) - 1
                else:
                    if info_dict['neuron_class'] in self.neurons:
                        neuron_name = info_dict['neuron_class']
                        neuron_id_per_dataset[dataset_name][neuron_name] = \
                                int(n_id) - 1

        return neuron_id_per_dataset

    def assemble_augmented_data(
            self,
            num_to_augment,
            noise_multiplier):
        """ combine original traces and behaviors with their augmentations """

        if num_to_augment == 0:
            return self.assemble_neural_behavior_data()
        elif num_to_augment > 0:
            orignal_data = self.assemble_neural_behavior_data()
            augmented_data = self._augment(num_to_augment, noise_multiplier)
            return np.vstack((orignal_data, augmented_data))
        else:
            ValueError("num to augment cannot be negative")

    def assemble_neural_behavior_data(self,):
        """ neural activities and behaviors without Gaussian noise """

        assembled_dataset = []
        for dataset_path in self.dataset_paths:

            dataset_name = dataset_path.split("/")[-1].split('.')[0]
            self.neuron_columns[dataset_name] = []
            self.behavior_columns[dataset_name] = []
            id_dict = self.neuron_id_per_dataset[dataset_name]

            with open(dataset_path, "r") as f:
                data = json.load(f)
                trace = np.array(data["trace_array"], dtype=np.float32).T

            # how to assemble dataset from `take_columns`
            # |take_columns| < |neurons| + |behaviors|
            # for missing neuron(s), we fill the column(s) with zeros
            # for this user case:
            #   take_all = True;
            #   |neurons| > 0; |behaviors| > 0

            # |take_column| = |neurons| + |behaviors|
            # there will be no missing neurons
            # for this user case:
            #   take_all = False;
            all_columns = []
            # assembled dataset contains 
            if self.take_all:
                for i, neuron in enumerate(self.neurons):
                    if neuron in id_dict.keys():
                        neuron_id = id_dict[neuron]
                        all_columns.append(self._normalize(trace[:1600,
                                                                 neuron_id]))
                        self._update_neuron_column(dataset_name,
                                                   len(all_columns)-1)
                    else:
                        all_columns.append(np.zeros(1600,))
                for i, behavior in enumerate(self.behaviors):
                    all_columns.append(self._normalize(np.array(data[behavior],
                                                                dtype=np.float32)[:1600]))
                    self._update_behavior_column(dataset_name,
                                                 len(all_columns)-1)
            else:
                for i, neuron in enumerate(self.neurons):
                    neuron_id = id_dict[neuron]
                    all_columns.append(self._normalize(trace[:1600,
                                                             neuron_id]))
                    self._update_neuron_column(dataset_name,
                                               len(all_columns)-1)

                for i, behavior in enumerate(self.behaviors):
                    all_columns.append(self._normalize(np.array(data[behavior],
                                                                dtype=np.float32)[:1600]))
                    self._update_behavior_column(dataset_name,
                                                 len(all_columns)-1)

            assembled_dataset.append(np.array([*all_columns]).T)

        return np.stack(assembled_dataset, axis=0)

    def _update_neuron_column(self, dataset_name, column_index):
        if column_index not in self.neuron_columns.values():
            self.neuron_columns[dataset_name].append(column_index)

    def _update_behavior_column(self, dataset_name, column_index):
        if column_index not in self.behavior_columns.values():
            self.behavior_columns[dataset_name].append(column_index)

    def augment_with_gaussian_noise(self, inputs, label):

        # inputs has shape (1, 1600, d)
        gfp_dataset_name = random.choice(
                ["2022-01-07-03", "2022-03-16-01", "2022-03-16-02"])
        gfp_dataset_path = \
        f"/home/alicia/store1/alicia/transformer/GFP/{gfp_dataset_name}.json"

        with open(gfp_dataset_path, "r") as f:
            data = json.load(f)
            gfp_stdev = np.std(np.array(data["trace_array"],
                                        dtype=np.float32).T)
        """
        augmented_inputs = copy.deepcopy(inputs)
        for col in self.neuron_columns[label]:
            augmented_inputs[0, :, col] = self._normalize(
                    inputs[0, :, col] + torch.tensor(
                        np.random.normal(0, self.noise_multiplier, (1600,)),
                    device=self.device))

        return augmented_inputs

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
                neuron_id = random.choice(list(id_dict.values()))
            elif len(id_dict) == 1:
                neuron_id = list(id_dict.values())[0]

            if not shuffle_animal:
                with open(dataset_path, "r") as f:
                    data = json.load(f)
                    trace = np.array(data["trace_array"], dtype=np.float32).T
                    behavior = np.array(data["velocity"], dtype=np.float32)

                AVA = trace[:1600, neuron_id]

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
                    AVA = trace[:1600, neuron_id]

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

    def _augment(self, num_to_augment, noise_multiplier):
        """ augment a chosen number of datasets by adding noise """

        assembled_dataset = []
        dataset_paths = random.sample(self.dataset_paths, num_to_augment)
        gfp_dataset_name = random.choice(
                ["2022-01-07-03", "2022-03-16-01", "2022-03-16-02"])
        gfp_dataset_path = \
        f"/home/alicia/store1/alicia/transformer/GFP/{gfp_dataset_name}.json"

        with open(gfp_dataset_path, "r") as f:
            data = json.load(f)
            gfp_stdev = np.std(np.array(data["trace_array"],
                                        dtype=np.float32).T)

        for dataset_path in dataset_paths:

            with open(dataset_path, "r") as f:
                data = json.load(f)
                trace = np.array(data["trace_array"], dtype=np.float32).T
                behavior = np.array(data["velocity"], dtype=np.float32)

            if behavior.shape[0] < 1600:
                continue

            dataset_name = dataset_path.split("/")[-1].split('.')[0]
            id_dict = self.neuron_id_per_dataset[dataset_name]
            if len(id_dict) == 2:
                AVA_id = random.choice(list(id_dict.values()))
            elif len(id_dict) == 1:
                AVA_id = list(id_dict.values())[0]

            AVA = self._normalize(trace[:1600, AVA_id] + \
                    noise_multiplier * np.random.normal(0, gfp_stdev, (1600,)))
            velocity = self._normalize(behavior[:1600])
            assembled_dataset.append(np.array([AVA, velocity]).T)

        return np.stack(assembled_dataset, axis=0)

    def _normalize(self, data):
        return (data - data.min()) / (data.max() - data.min()) * 2 - 1

    def __getitem__(self, index):
        x = self.input_embeddings[index]
        y = self.target_embeddings[index]
        label = self.labels[index]
        return x, y, label

    def __len__(self):
        return len(self.input_embeddings)


def test():
    dataset_paths = glob.glob(f"/storage/fs/store1/alicia/transformer/AVA/*.json")
    data_parameters = DataParameters(
            dataset_paths,
            neurons = ["AVA"],
            behaviors = ["velocity"],
            noise_multiplier = 0.12,
            num_to_augment = 0,
            take_all = False,
            ignore_LRDV = True,
            device = "cuda:3")
    dataset = WormDataset(data_parameters)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True)
    print(len(dataloader.dataset))
    for i, (inputs, targets, label) in enumerate(dataloader):
        #print(f"augmented_inputs: {augmented_inputs.shape}")
        print(f"batch {i}, dataset: {label} inputs: {inputs.shape} targets: {targets.shape}\n")
        #augmented_inputs = dataset.augment_with_gaussian_noise(inputs)

if __name__ == "__main__":
    test()

