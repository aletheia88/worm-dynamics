from attention_predict.reconstruct_mini import (
        reconstruct_traces,
        build_mini_attention_model,
        build_dataloader
)
from tqdm import tqdm
import attention_predict
import json
import numpy as np
import torch


def construct_valid_loss_curve(
    attention_scheme,
    ds_name,
    ckpts,
    ckpt_path,
    device
):
    depth = 5
    num_fmaps = 64
    num_neurons = 70
    num_behaviors = 4
    num_datasets = 5
    window_size = 400

    # load training losses
    loss_dict = {'training': [], 'validation': []}
    # TODO: replace losses.json with losses_all.json
    with open(f'{ckpt_path}/losses.json', 'r') as f:
        losses = json.load(f)
        loss_dict['training'] = losses['training']
        loss_dict['validation'] = losses['validation']

    # initialize dataloader
    valid_dataloader = build_dataloader(ds_name, device)

    for n_ckpt in ckpts:
        print(f'evaluating with ckpt {n_ckpt}...')
        loss = estimate_loss(
            valid_dataloader,
            attention_scheme,
            depth,
            num_fmaps,
            num_neurons,
            num_behaviors,
            num_datasets,
            window_size,
            ckpt_path,
            n_ckpt,
            device
        )
        loss_dict['validation'].append(loss)

    # write new json file that includes the validation loss
    with open(f'{ckpt_path}/losses_all.json', 'w') as f:
        json.dump(loss_dict, f, indent=4)


def estimate_loss(
    valid_dataloader,
    attention_scheme,
    depth,
    num_fmaps,
    num_neurons,
    num_behaviors,
    num_datasets,
    window_size,
    ckpt_path,
    n_ckpt,
    device
):
    # initialize model
    model = build_mini_attention_model(
        attention_scheme,
        depth,
        num_neurons,
        num_behaviors,
        window_size,
        num_fmaps,
        device
    )
    print('attention model built!')
    # load checkpoint
    checkpoint = torch.load(
            f'{ckpt_path}/model_ckpt{n_ckpt}.pt',
            map_location=device
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # evaluate model on the validation set
    validation_loss = []

    # get indices to model inputs and outputs
    neuron_indices = list(range(num_neurons))
    behavior_indices = list(range(num_neurons, num_neurons+num_behaviors))

    if attention_scheme == 'BfromN':
        input_indices = neuron_indices
        output_indices = behavior_indices

    elif attention_scheme == 'NfromB':
        input_indices = behavior_indices
        output_indices = neuron_indices

    elif attention_scheme == 'NfromN':
        input_indices = neuron_indices
        output_indices = neuron_indices

    elif attention_scheme == 'BfromB':
        input_indices = behavior_indices
        output_indices = behavior_indices

    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss()

    with torch.no_grad():

        for inputs, _, _, _, _ in tqdm(valid_dataloader):
            outputs, _ = model(inputs[:, input_indices, :])
            targets = inputs[:, output_indices, :]

            if attention_scheme == 'BfromN':
                loss = criterion(targets, outputs)

            if attention_scheme in ['NfromB', 'NfromN']:
                num_samples, _, window_size = targets.shape
                loss_mask = torch.zeros(
                        (num_samples, num_neurons, window_size),
                        dtype=bool)
                recorded_neuron_indices = get_recorded_neurons(
                        targets,
                        num_neurons)
                for n, loss_indices in recorded_neuron_indices.items():
                    loss_mask[n, loss_indices, :] = True

                loss = criterion(targets[loss_mask], outputs[loss_mask])

            validation_loss.append(loss.item())

    return np.mean(validation_loss)


def get_recorded_neurons(targets, num_neurons):

    num_samples = targets.shape[0]
    recorded_neuron_indices = {}

    for n in range(num_samples):
        recorded_neuron_indices[n] = [
            i for i in range(num_neurons)
            if (torch.max(targets[n, i, :]).item() != -10
            and torch.min(targets[n, i, :]).item() != -10)
        ]

    return recorded_neuron_indices


if __name__ == "__main__":

    import os
    import re

    ds_name = 'data0410_norm0505_valid'
    # ckpts = list(range(1, 403, 2))
    device = 'cuda:1'
    attention_scheme = 'NfromB'

    base = '/store1/alicia/attention_predict/whole_brain'
    # noise = '0.5'
    ckpt_path = f'{base}/depth5_nfmaps64_lr5e-05_{attention_scheme}_norm0505_MAE'
    # List all files in the directory
    files = os.listdir(ckpt_path)

    # Extract checkpoint numbers using regex
    checkpoints = sorted([
        int(match.group(1)) for fname in files
        if (match := re.match(r"model_ckpt(\d+)\.pt", fname))
    ])
    print(checkpoints)

    construct_valid_loss_curve(
        attention_scheme,
        ds_name,
        checkpoints,
        ckpt_path,
        device
    )
