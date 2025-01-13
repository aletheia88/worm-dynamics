from attention_predict.dataset import CElegansDatasetPlus
from attention_predict.attention_model_3 import AttentionModel3
from attention_predict.attention_model_2 import AttentionModel2
from attention_predict.attention_model import AttentionModel
from copy import deepcopy
from tqdm import tqdm
import json
import numpy as np
import torch


def train(
    num_neurons,
    training_dataset,
    model,
    attention_scheme,
    batch_size,
    num_iterations,
    num_epochs,
    learning_rate,
    random_seed,
    log_directory,
    log_ckpt_freq=None,
):

    ckpt_directory = f'{log_directory}/checkpoints'
    ensure_dir_exists([ckpt_directory])

    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = torch.nn.MSELoss()

    loss_dict = {'training': [], 'validation': []}
    num_inputs = next(iter(training_dataloader))[0].shape[1]

    neuron_indices = list(range(num_neurons))
    behavior_indices = list(range(num_neurons, num_inputs))

    # set random seed for reproducible masking
    # torch.manual_seed(random_seed)
    # np.random.seed(random_seed)

    if attention_scheme in ['BfromN', 'BfromB']:
        loss_indices = behavior_indices

    for n_epoch in tqdm(range(num_epochs)):

        for n_iter, (inputs, _, _, _, _) in tqdm(enumerate(training_dataloader)):

            if n_iter == num_iterations:
                break

            targets = deepcopy(inputs)
            num_samples = targets.shape[0]

            if attention_scheme in ['NfromN', 'NfromB']:
                recorded_neuron_indices = get_recorded_neuron_indices(targets, num_neurons,
                                                           num_samples)
                # randomly drop one recorded neuron
                loss_indices = drop_neuron(recorded_neuron_indices, num_samples, 1)

            optimizer.zero_grad()
            outputs, attention_weights = model(inputs)
            loss = aggregate_loss(
                    targets,
                    outputs,
                    loss_indices,
                    mse_loss,
                    num_samples,
                    attention_scheme,
                    num_neurons,
            )
            loss.backward()
            # add gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # log average loss over all samples in the batch per iteration
            loss_dict['training'].append(loss.item())

        with open(f'{log_directory}/losses.json', 'w') as f:
            json.dump(loss_dict, f, indent=4)

        if log_ckpt_freq is not None and n_epoch % log_ckpt_freq == 0:

            torch.save(
                {
                    'epoch': n_epoch,
                    'state_dict': model.state_dict(),
                    'attention': attention_weights,
                    'optimizer': optimizer.state_dict(),
                },
                    f'{log_directory}/checkpoints/model_ckpt{n_epoch}.pt'
            )


def ensure_dir_exists(directories):
    import os
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def aggregate_loss(
    targets,
    outputs,
    loss_indices,
    mse_loss,
    num_samples,
    attention_scheme,
    num_neurons,
):
    """ Compute MSE loss for each batch separately on the reconstruction of recorded
    neural and behavioral activities. """

    loss = 0
    if attention_scheme in ['NfromN', 'NfromB']:
        for n in range(num_samples):
            for neuron_index in loss_indices[n]:
                loss += mse_loss(
                    targets[n, neuron_index, :],
                    outputs[n, neuron_index, :]
                )
    elif attention_scheme in ['BfromN', 'BfromB']:
        for behavior_index in loss_indices[:3]:
            loss += mse_loss(
                    targets[:, behavior_index, :],
                    outputs[:, behavior_index, :]
            )
        loss += mse_loss(
            targets[:, num_neurons+3:-1, :],
            outputs[:, num_neurons+3:-1, :]
        )

    return loss


def get_recorded_neuron_indices(targets, num_neurons, num_samples):

    recorded_neuron_indices = {}

    for n in range(num_samples):
        recorded_neuron_indices[n] = [
            i for i in range(num_neurons)
            if (torch.max(targets[n, i, :]).item() != -10
            and torch.min(targets[n, i, :]).item() != -10)
        ]
    return recorded_neuron_indices


def drop_neuron(recorded_neuron_indices, num_samples, num_drop):

    loss_indices = recorded_neuron_indices.copy()

    for n in range(num_samples):
        drop_index = np.random.choice(recorded_neuron_indices[n], num_drop)
        loss_indices.remove(drop_index)

    return loss_indices


if __name__ == '__main__':

    device = "cuda:2"
    window_size = 400
    window_stride = 1
    ds_name = 'data0108_norm'
    batch_size = 32
    base = '/home/alicia/store1/alicia/attention_predict'

    training_dataset = CElegansDatasetPlus(
        f'{base}/data/{ds_name}_train.npy',
        f'{base}/data/{ds_name}_train_ds.npy',
        window_stride=window_stride,
        window_size=window_size,
        device=device,
        slices=slice(0, 1600)
    )
    num_iterations = 1_000_000 #1_000_000
    num_epochs = 2001
    learning_rate = 1e-5
    exp_name = 'exp_2025010801'
    log_directory = f'{base}/{exp_name}'
    log_ckpt_freq = 10
    random_seed = 1912 # Alan Turing's birth year :)

    depth = 3
    num_inputs = 51
    num_neurons = 17
    num_behaviors = 34 # 3 (cepnem beh) + 30 (body angles) + 1 (heat-stim)
    attention_scheme = 'NfromB'

    model = AttentionModel2(
        depth,
        num_neurons,
        num_behaviors,
        window_size,
        attention_scheme=attention_scheme,
        device=device
    )
    train(
        num_neurons,
        training_dataset,
        model,
        attention_scheme,
        batch_size,
        num_iterations,
        num_epochs,
        learning_rate,
        random_seed,
        log_directory,
        log_ckpt_freq
    )

