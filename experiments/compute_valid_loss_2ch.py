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
    device,
    depth,
    num_fmaps,
    perturb_index,
):
    num_neurons = 70
    num_behaviors = 4
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
            window_size,
            ckpt_path,
            n_ckpt,
            perturb_index,
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
    window_size,
    ckpt_path,
    n_ckpt,
    perturb_index,
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
        perturb_index,
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

    num_inputs = num_neurons + num_behaviors
    # get indices to model inputs and outputs
    neuron_indices = list(range(num_neurons * 2))
    behavior_indices = list(range(num_neurons * 2, num_inputs * 2))

    if attention_scheme == 'BfromN':
        input_indices = neuron_indices
        output_indices = behavior_indices

    elif attention_scheme == 'NfromB':
        input_indices = behavior_indices
        output_indices = neuron_indices

    elif attention_scheme in ['NfromN', 'perturb']:
        input_indices = neuron_indices
        output_indices = neuron_indices

    elif attention_scheme == 'BfromB':
        input_indices = behavior_indices
        output_indices = behavior_indices

    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.L1Loss()

    with torch.no_grad():

        for inputs, _, _, _, _ in tqdm(valid_dataloader):
            outputs = model(inputs[:, input_indices, :])
            targets = inputs[:, output_indices, :]

            recorded_neuron_indices = get_recorded_neurons(targets)
            loss = aggregate_loss(
                    targets,
                    outputs,
                    criterion,
                    attention_scheme,
                    recorded_neuron_indices)

            validation_loss.append(loss.item())

    return np.mean(validation_loss)


def aggregate_loss(
    targets,
    outputs,
    criterion,
    attention_scheme,
    recorded_neuron_indices
):
    # ouputs: (num_samples, num_variables, window_size)
    # targets: (num_samples, num_variables * 2, window_size)
    num_samples, num_vecs, window_size = targets.shape
    num_variables = num_vecs // 2

    if attention_scheme == 'BfromN':
        # E.g.,
        # target indices: 0, 1 | 2, 3 | 4, 5
        # where 0, 2, 4 index into acitvity recordings
        # output indices: 0, 1, 2
        loss_indices = list(range(0, num_vecs, 2))
        return criterion(outputs, targets[:, loss_indices, :])

    if attention_scheme in ['NfromN', 'NfromB', 'connectome', 'anticonnectome']:
        if recorded_neuron_indices is None:
            raise ValueError(
                    'Function input recorded_neurons_indices cannot be None.'
            )
        target_loss_mask = torch.zeros(
                (num_samples, num_vecs, window_size),
                dtype=bool)
        output_loss_mask = torch.zeros(
                (num_samples, num_variables, window_size),
                dtype=bool)

        for n, loss_indices in recorded_neuron_indices.items():
            target_loss_mask[n, loss_indices, 20:380] = True
            output_loss_mask[n, [i//2 for i in loss_indices], 20:380] = True

        return criterion(outputs[output_loss_mask], targets[target_loss_mask])


def get_recorded_neurons(targets):

    recorded_neuron_indices = {}
    num_samples, num_vecs, _ = targets.shape
    # num_vecs = num_variables to predicted * 2

    for n in range(num_samples):
        recorded_neuron_indices[n] = [
            i - 1
            for i in range(1, num_vecs, 2)
            if targets[n, i, 0] == 1
        ]

    return recorded_neuron_indices


if __name__ == "__main__":

    import os
    import re

    # ds_name = 'data0626/data0626-00e_norm0505_valid'
    ds_name = 'data0715e_norm0505_valid'
    # ckpts = list(range(1, 403, 2))
    device = 'cuda:2'
    attention_scheme = 'NfromB'

    base = '/store1/alicia/attention_predict/whole_brain'
    d = 5
    f = 64
    perturb_index = None
    # noise = '0.5'
    exp_name = 'currbest_fix'
    ckpt_path = f'{base}/depth{d}_nfmaps{f}_lr5e-05_{attention_scheme}-4B_norm0505_{exp_name}'
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
        device,
        d,
        f,
        perturb_index,
    )
