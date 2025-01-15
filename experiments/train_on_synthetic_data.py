from attention_predict.attention_model_2 import AttentionModel2
from attention_predict.dataset import CElegansDatasetPlus
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch


def create_data_files():

    num_datasets = 100
    num_inputs = 5
    max_length = 400
    amplitude_range = (1, 10)

    synthetic_data = generate_synthetic_data(
        num_datasets,
        num_inputs,
        max_length,
        amplitude_range
    )
    ds_name = 'synthetic'
    train_set_ratio = 0.7
    train_set, test_set = split_train_test(
        synthetic_data,
        num_datasets,
        train_set_ratio,
        ds_name
    )
    normalized_train_data, zscore_params = zscore_normalize(train_set)
    print(zscore_params)
    normalized_test_data = zscore_normalize(test_set, zscore_params)

    data_dir = '/store1/alicia/attention_predict/data'
    np.save(f'{data_dir}/{ds_name}_train_norm.npy', normalized_train_data)
    np.save(f'{data_dir}/{ds_name}_eval_norm.npy', normalized_test_data)


def generate_synthetic_data(num_datasets, num_inputs, max_length, amplitude_range):

    amplitude_scalers = [0.1 * (i+1) for i in range(num_inputs)]
    all_samples = []

    for _ in range(num_datasets):

        single_sample = []
        for i in range(num_inputs):

            time = np.linspace(0, 100, max_length)
            amplitude = np.random.uniform(*amplitude_range)
            frequency = np.random.uniform(0.5, 2.0)
            phase_shift = np.random.uniform(0, 2 * np.pi)
            noise = np.random.normal(0, 0.01 * amplitude, max_length)

            values = amplitude_scalers[i] * amplitude * np.sin(frequency * time + phase_shift) + noise
            single_sample.append(values)

        all_samples.append(np.array(single_sample))

    return np.array(all_samples)


def split_train_test(data, num_datasets, train_set_ratio, ds_name, save=False):

    data_dir = '/store1/alicia/attention_predict/data'
    train_size = int(num_datasets * train_set_ratio)
    test_size = num_datasets - train_size
    print(f'train-test splits: {train_size, test_size}')

    indices = np.random.choice(num_datasets, num_datasets, replace=False)

    train_indices = indices[:train_size]
    test_indices = indices[-test_size:]

    train_set = data[train_indices, :, :]
    test_set = data[test_indices, :, :]

    if save:
        np.save(f'{data_dir}/{ds_name}_raw_train.npy', train_set)
        np.save(f'{data_dir}/{ds_name}_raw_eval.npy', test_set)
        print(f'train-test splits for {ds_name}_raw saved!')

    return train_set, test_set


def create_data_files():

    num_datasets = 100
    num_inputs = 5
    max_length = 400
    amplitude_range = (1, 10)

    synthetic_data = generate_synthetic_data(
        num_datasets,
        num_inputs,
        max_length,
        amplitude_range
    )
    ds_name = 'synthetic'
    train_set_ratio = 0.7
    train_set, test_set = split_train_test(
        synthetic_data,
        num_datasets,
        train_set_ratio,
        ds_name
    )
    normalized_train_data, zscore_params = zscore_normalize(train_set)
    print(zscore_params)
    normalized_test_data = zscore_normalize(test_set, zscore_params)

    data_dir = '/store1/alicia/attention_predict/data'
    np.save(f'{data_dir}/{ds_name}_train_norm.npy', normalized_train_data)
    np.save(f'{data_dir}/{ds_name}_eval_norm.npy', normalized_test_data)


if __name__ == '__main__':
    create_data_files()
