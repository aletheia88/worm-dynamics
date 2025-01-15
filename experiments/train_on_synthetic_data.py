from attention_predict.attention_model_2 import AttentionModel2
from attention_predict.dataset import CElegansDatasetPlus
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch


def train(mse_scheme):

    depth = 3
    num_neurons = 5
    num_behaviors = 0 # not used
    attention_scheme = 'NfromN'
    device = "cuda:2"
    data_dir = '/home/alicia/store1/alicia/attention_predict/data'
    ds_name = 'synthetic'
    window_size = 400
    window_stride = 1
    batch_size = 32

    model = AttentionModel2(
        depth,
        num_neurons,
        num_behaviors,
        window_size,
        attention_scheme=attention_scheme,
        device=device
    )
    training_dataset = CElegansDatasetPlus(
        f'{data_dir}/{ds_name}_train_norm.npy',
        f'{data_dir}/{ds_name}_train_ds.npy',
        window_stride=window_stride,
        window_size=window_size,
        device=device,
        slices=slice(0, 1600)
    )
    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    learning_rate = 1e-5
    num_epochs = 500
    num_iterations = 1_000_000
    log_dir = f'/home/alicia/store1/alicia/attention_predict/exp_{mse_scheme}'
    ensure_dir_exists([f'{log_dir}/checkpoints'])

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = torch.nn.MSELoss(reduction='sum')

    loss_dict = {'training': [], 'validation': []}
    num_inputs = next(iter(training_dataloader))[0].shape[1]
    loss_indices = list(range(num_neurons))

    for n_epoch in tqdm(range(num_epochs)):

        for n_iter, (inputs, _, _, _, _) in tqdm(enumerate(training_dataloader)):

            if n_iter == num_iterations:
                break

            num_samples = inputs.shape[0]
            optimizer.zero_grad()
            outputs, attention_weights = model(inputs)
            loss = compute_loss(inputs, outputs, mse_loss, mse_scheme)
            loss.backward()
            optimizer.step()
            loss_dict['training'].append(loss.item())

        with open(f'{log_dir}/losses.json', 'w') as f:
            json.dump(loss_dict, f, indent=4)

        if log_ckpt_freq is not None and n_epoch % log_ckpt_freq == 0:

            torch.save(
                {
                    'epoch': n_epoch,
                    'state_dict': model.state_dict(),
                    'attention': attention_weights,
                    'optimizer': optimizer.state_dict(),
                },
                    f'{log_dir}/checkpoints/model_ckpt{n_epoch}.pt'
            )


def ensure_dir_exists(directories):
    import os
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def compute_loss(targets, outputs, mse_loss, mse_scheme):

    num_samples, num_variables, length = targets.shape

    if mse_scheme == 'average_by_length':
        loss = mse_loss(targets, outputs) / length

    elif mse_scheme == 'average_by_variable':
        loss = mse_loss(targets, outputs) / num_variables

    elif mse_scheme == 'average_by_all':
        loss = mse_loss(targets, outputs) / (length * num_variables)

    elif mse_scheme == 'sum':
        loss = mse_loss(targets, outputs)

    return loss / num_samples


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
    train('average_by_length')
