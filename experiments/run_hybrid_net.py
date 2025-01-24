from attention_predict.dataset import CElegansDatasetPlus
from attention_predict.hybrid_net import HybridNet
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
    reconstruction_loss = torch.nn.MSELoss()

    loss_dict = {'training': [], 'validation': []}
    num_inputs = next(iter(training_dataloader))[0].shape[1]

    neuron_indices = list(range(num_neurons))
    behavior_indices = list(range(num_neurons, num_inputs))

    # set random seed for reproducible masking
    # torch.manual_seed(random_seed)
    # np.random.seed(random_seed)

    if attention_scheme in ['BfromN', 'BfromB']:
        loss_indices = {n: behavior_indices for n in range(batch_size)}

    for n_epoch in tqdm(range(num_epochs)):

        for n_iter, (inputs, _, _, _, _) in tqdm(enumerate(training_dataloader)):

            if n_iter == num_iterations:
                break

            targets = deepcopy(inputs)
            num_samples = targets.shape[0]

            if attention_scheme in ['NfromN', 'NfromB']:
                recorded_neuron_indices = get_recorded_neuron_indices(targets, num_neurons)
                mask_indices = get_mask_indices(
                        num_inputs,
                        recorded_neuron_indices,
                        neuron_indices)
                # zeroing out the missing neurons
                for n in mask_indices.keys():
                    inputs[n, mask_indices[n], :] = 0

            optimizer.zero_grad()
            outputs, attention_weights = model(inputs)

            # compute MSE loss based on attention scheme
            if attention_scheme in ['NfromN', 'NfromB']:
                loss_indices = recorded_neuron_indices

            loss = aggregate_loss(
                    targets,
                    outputs,
                    loss_indices,
                    reconstruction_loss,
                    num_samples)
            loss.backward()
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


def get_mask_indices(
    num_inputs,
    recorded_neuron_indices,
    neuron_indices,
):
    """ Masking out the missing neurons """
    mask_indices = {}
    for n in recorded_neuron_indices.keys():
        mask_indices[n] = list(set(neuron_indices) - set(recorded_neuron_indices[n]))

    return mask_indices


def aggregate_loss(
    targets,
    outputs,
    loss_indices,
    reconstruction_loss,
    num_samples
):
    """ Compute MSE loss for each batch separately on the reconstruction of recorded
    neural and behavioral activities. """

    loss = 0
    for n in range(num_samples):
        loss += reconstruction_loss(targets[n, loss_indices[n], :],
                                    outputs[n, loss_indices[n], :])
    return loss / num_samples


def get_recorded_neuron_indices(targets, num_neurons):

    batch_size = targets.shape[0]
    recorded_neuron_indices = {}

    for n in range(batch_size):
        recorded_neuron_indices[n] = [
            i for i in range(num_neurons)
            if (torch.max(targets[n, i, :]).item() != 0
            and torch.min(targets[n, i, :]).item() != 0)
        ]
    return recorded_neuron_indices


if __name__ == '__main__':

    device = "cuda:2"
    window_size = 400
    window_stride = 1
    ds_name = 'AVA_MC_SMDV'
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
    num_epochs = 1000
    learning_rate = 1e-4
    exp_name = 'exp_2024111901'
    log_directory = f'{base}/{exp_name}'
    log_ckpt_freq = 10
    random_seed = 1912 # Alan Turing's birth year :)

    depth = 5
    num_inputs = 6
    num_neurons = 3
    num_behaviors = 3
    attention_scheme = 'BfromN'

    model = HybridNet(
            depth,
            num_neurons,
            num_behaviors,
            window_size,
            attention_scheme=attention_scheme,
            device=device)
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
        log_ckpt_freq)

