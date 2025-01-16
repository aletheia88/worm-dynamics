from attention_predict.dataset import CElegansDatasetPlus
from attention_predict.reconstruct import reconstruct_traces, build_model, build_dataloader
from copy import deepcopy
from tqdm import tqdm
import json
import numpy as np
import torch


# Approach to evaluate model
# 1. get attention matrix and normalize according to the baseline
# 2. for each row, rank the attention scores of each behavior from high to low
# 3. get the top k behaviors (and their column indices) with the highest attention score
# 4. delete each behavior from inputs and get model outputs
# 5. compute MSE which contributes a dot on the final violin plot

def evaluate_signal_mixing(
    num_neurons,
    num_behaviors,
    num_worms,
    num_top_contributors,
    contribution_rank,
    dataloader,
    model,
    attention_scheme
):
    max_length = 1600
    window_size = 400
    num_windows = max_length // window_size

    reconstruction_loss = torch.nn.MSELoss(reduction='sum')
    all_indices = list(range(num_neurons + num_behaviors))
    dataset = dataloader.dataset

    evaluation = {}
    context_ids = create_context_ids(num_top_contributors)

    if attention_scheme == 'BfromN':
        variable_indices = list(range(num_neurons, num_neurons + num_behaviors))
        mix_signals = build_B_from_Ns

    elif attention_scheme == 'NfromB':
        variable_indices = list(range(num_neurons))
        mix_signals = build_N_from_Bs

    elif attention_scheme == 'NfromN':
        variable_indices = list(range(num_neurons))
        mix_signals = build_N_from_Ns

    for variable_index in tqdm(variable_indices):

        evaluation[variable_index] = {context_id: [] for context_id in context_ids}

        for worm_index in range(num_worms):

            reconstruction_mse = mix_signals(
                num_top_contributors,
                contribution_rank,
                num_windows,
                num_neurons,
                model,
                dataset,
                all_indices,
                worm_index,
                variable_index,
                reconstruction_loss,
            )
            # if this worm dataset has neuron_index recorded
            if reconstruction_mse:
                for context_id, mse in reconstruction_mse.items():
                    if np.array(mse).size > 0:
                        evaluation[variable_index][context_id].append(mse)

    return evaluation


@torch.no_grad()
def build_N_from_Bs(
    num_top_contributors,
    contribution_rank,
    num_windows,
    num_neurons,
    model,
    dataset,
    all_indices,
    worm_index,
    neuron_index,
    reconstruction_loss,
):
    end_frame_index = (worm_index + 1) * num_windows - 1
    start_frame_index = end_frame_index - 3
    # check if given neuron_index is recorded in dataset worm_index
    recorded_neuron_indices = get_recorded_neuron_indices(
        dataset[start_frame_index][0].unsqueeze(0),
        num_neurons
    )
    if neuron_index not in recorded_neuron_indices:
        return False

    neuron_from_behaviors = {
        neuron_index: behaviors[:num_top_contributors]
        for neuron_index, behaviors in contribution_rank.items()
    }
    behavior_indices = neuron_from_behaviors[neuron_index]
    contexts = establish_contexts(behavior_indices)

    reconstruction_mse = {}

    for context_id, context_indices in contexts.items():

        reconstruction_mse[context_id] = []
        ignore_indices = list(set(all_indices) - set(context_indices))

        for window_index in range(start_frame_index, end_frame_index + 1):

            inputs = dataset[window_index][0].unsqueeze(0)
            targets = deepcopy(inputs)
            # set all indices except context_indices to 0 (mean activity)
            inputs[:, ignore_indices, :] = 0
            outputs, _ = model(inputs)
            loss = reconstruction_loss(
                targets[:, neuron_index, :],
                outputs[:, neuron_index, :]
            ).item()
            reconstruction_mse[context_id].append(loss)

        reconstruction_mse[context_id] = np.sum(reconstruction_mse[context_id])

    return reconstruction_mse


@torch.no_grad()
def build_B_from_Ns(
    num_top_contributors,
    contribution_rank,
    num_windows,
    num_neurons,
    model,
    dataset,
    all_indices,
    worm_index,
    behavior_index,
    reconstruction_loss,
):
    end_frame_index = (worm_index + 1) * num_windows - 1
    start_frame_index = end_frame_index - 3

    recorded_neuron_indices = get_recorded_neuron_indices(
        dataset[start_frame_index][0].unsqueeze(0),
        num_neurons
    )

    behavior_from_neurons = {
        behavior_index: neurons[:num_top_contributors]
        for behavior_index, neurons in contribution_rank.items()
    }
    neuron_indices = behavior_from_neurons[behavior_index]
    contexts = establish_contexts(neuron_indices)

    reconstruction_mse = {}

    for context_id, context_indices in contexts.items():

        reconstruction_mse[context_id] = []

        # check if context_indices contain neurons that are not recorded
        if not all(num in recorded_neuron_indices for num in context_indices):
            continue

        ignore_indices = list(set(all_indices) - set(context_indices))

        for window_index in range(start_frame_index, end_frame_index + 1):

            inputs = dataset[window_index][0].unsqueeze(0)
            targets = deepcopy(inputs)
            # set all indices except context_indices to 0 (mean activity)
            inputs[:, ignore_indices, :] = 0
            outputs, _ = model(inputs)
            loss = reconstruction_loss(
                targets[:, behavior_index, :],
                outputs[:, behavior_index, :]
            ).item()
            reconstruction_mse[context_id].append(loss)

        reconstruction_mse[context_id] = np.sum(reconstruction_mse[context_id])

    return reconstruction_mse


@torch.no_grad()
def build_N_from_Ns(
    num_top_contributors,
    contribution_rank,
    num_windows,
    num_neurons,
    model,
    dataset,
    all_indices,
    worm_index,
    neuron_index,
    reconstruction_loss,
):
    end_frame_index = (worm_index + 1) * num_windows - 1
    start_frame_index = end_frame_index - 3

    recorded_neuron_indices = get_recorded_neuron_indices(
        dataset[start_frame_index][0].unsqueeze(0),
        num_neurons
    )
    if neuron_index not in recorded_neuron_indices:
        return False

    neuron_from_neurons = {
        neuron_index: neurons[:num_top_contributors]
        for neuron_index, neurons in contribution_rank.items()
    }
    neuron_indices = neuron_from_neurons[neuron_index]
    contexts = establish_contexts(neuron_indices)

    reconstruction_mse = {}
    for context_id, context_indices in contexts.items():

        reconstruction_mse[context_id] = []

        # check if context_indices contain neurons that are not recorded
        if not all(num in recorded_neuron_indices for num in context_indices):
            continue

        ignore_indices = list(set(all_indices) - set(context_indices))

        for window_index in range(start_frame_index, end_frame_index + 1):

            inputs = dataset[window_index][0].unsqueeze(0)
            targets = deepcopy(inputs)
            # set all indices except context_indices to 0 (mean activity)
            inputs[:, ignore_indices, :] = 0
            outputs, _ = model(inputs)
            loss = reconstruction_loss(
                targets[:, neuron_index, :],
                outputs[:, neuron_index, :]
            ).item()
            reconstruction_mse[context_id].append(loss)

        reconstruction_mse[context_id] = np.sum(reconstruction_mse[context_id])

    return reconstruction_mse


def rank_contributors(reconstruction, attention_scheme, N, B):
    # N = num_neurons
    if attention_scheme == 'NfromB':
        attn_matrix = reconstruction[0]['attn_weights'][0][0, :N, N:]
    elif attention_scheme == 'BfromN':
        attn_matrix = reconstruction[0]['attn_weights'][0][0, N:, :N]
    elif attention_scheme == 'NfromN':
        attn_matrix = reconstruction[0]['attn_weights'][0][0, :N, :N]

    baseline_attn = 1 / attn_matrix.shape[1]
    normalized_attn_matrix = (attn_matrix - baseline_attn) / baseline_attn

    if attention_scheme == 'NfromB':
        contribution_rank = {i: N + np.argsort(normalized_attn_matrix[i])[::-1] for i in range(N)}
    elif attention_scheme == 'BfromN':
        contribution_rank = {N + i: np.argsort(normalized_attn_matrix[i])[::-1] for i in range(B)}
    elif attention_scheme == 'NfromN':
        contribution_rank = {i: np.argsort(normalized_attn_matrix[i])[::-1] for i in range(N)}

    return contribution_rank


def load_model_weights(model, experiment, model_ckpt):

    model_ckpt_path = f'/store1/alicia/attention_predict/{experiment}/checkpoints/model_ckpt{model_ckpt}.pt'
    checkpoint = torch.load(model_ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return model


def create_context_ids(num_top_contributors):

    context_ids = [f"v{i}" for i in range(1, num_top_contributors + 1)]
    combinations = ["-".join([f"v{j}" for j in range(1, i + 1)])
                    for i in range(2, num_top_contributors + 1)]
    context_ids.extend(combinations)

    return context_ids


def get_recorded_neuron_indices(targets, num_neurons):

    return [
        i for i in range(num_neurons)
        if (torch.max(targets[0, i, :]).item() != -10
        and torch.min(targets[0, i, :]).item() != -10)
    ]


def establish_contexts(variable_indices):

    # variable_indices is ordered from high to low contribution
    # contexts that consist of singleton neural/behavioral variable
    context_index_list = [[variable_index] for variable_index in variable_indices]
    # contexts that consist of >= 2 neural/behavioral variables
    context_index_list += [variable_indices[:i] for i in range(2, len(variable_indices)+1)]

    contexts = {}
    # context_index_list: [[b1i], [b2i], [b3i], [b1i, b2i], [b1i, b2i, b3i]]
    # context_id: [v1, v2, v3, v1-v2, v1-v2-v3]
    for i, context_indices in enumerate(context_index_list):
        if len(context_indices) == 1:
            context_id = f'v{i+1}'
        else:
            context_id = '-'.join(f'v{i}' for i in range(1, len(context_indices)+1))
        contexts[context_id] = list(context_indices)

    return contexts


def print_contexts(num_top_contributors):

    architecture = 'attention_model_2'
    attention_scheme = 'BfromN'
    ds_name = 'data0108_norm_eval'
    device = 'cuda:2'
    model_ckpt = 760
    experiment = 'exp_2025011300'
    num_neurons = 17
    num_behaviors = 34
    num_worms = 20

    model = build_model(
        architecture,
        attention_scheme,
        num_neurons=num_neurons,
        num_behaviors=num_behaviors,
        device=device
    )
    dataloader = build_dataloader(ds_name, device)
    # model weights loaded inside the function reconstruct_traces
    reconstruction = reconstruct_traces(
        model,
        dataloader,
        model_ckpt,
        experiment,
        num_worms,
        architecture,
        num_neurons=num_neurons
    )
    contribution_rank = rank_contributors(
        reconstruction,
        attention_scheme,
        num_neurons,
        num_behaviors
    )
    neurons = ['SMDV', 'SMDD', 'SAADL', 'SAADR', 'SAAV',
               'MC', 'M3', 'M4', 'MI',
               'AVA', 'AVB', 'RIB',
               'RME', 'RMEV', 'RMED',
               'URYD', 'URYV']
    behaviors = ['velocity', 'pumping', 'head-angle']
    num_body_angles = 30
    behaviors += [f'body-angle-{i}' for i in range(1, num_body_angles+1)]
    behaviors += ['heat-stim']
    variables = neurons + behaviors

    contexts = {
        variables[variable_index]: [
            variables[rank_index]
            for rank_index in rank_indices[:num_top_contributors]
        ]
        for variable_index, rank_indices in contribution_rank.items()
    }
    print(f'Top {num_top_contributors} contributors: {contexts}')


def main():

    architecture = 'attention_model_2'
    attention_scheme = 'BfromN'
    ds_name = 'data0108_norm_eval'
    device = 'cuda:2'
    model_ckpt = 760
    experiment = 'exp_2025011300'
    num_neurons = 17
    num_behaviors = 34
    num_worms = 20

    model = build_model(
        architecture,
        attention_scheme,
        num_neurons=num_neurons,
        num_behaviors=num_behaviors,
        device=device
    )
    dataloader = build_dataloader(ds_name, device)
    # model weights loaded inside the function reconstruct_traces
    reconstruction = reconstruct_traces(
        model,
        dataloader,
        model_ckpt,
        experiment,
        num_worms,
        architecture,
        num_neurons=num_neurons
    )
    contribution_rank = rank_contributors(
        reconstruction,
        attention_scheme,
        num_neurons,
        num_behaviors
    )
    model = load_model_weights(model, experiment, model_ckpt)
    num_top_contributors = 3
    # num_windows = 4
    # worm_index = 17
    # neuron_index = 7
    # dataset = dataloader.dataset
    # all_indices = list(range(num_neurons + num_behaviors))
    # reconstruction_loss = torch.nn.MSELoss()

    # reconstruction_mse = build_N_from_Ns(
    #     num_top_contributors,
    #     contribution_rank,
    #     num_windows,
    #     num_neurons,
    #     model,
    #     dataset,
    #     all_indices,
    #     worm_index,
    #     neuron_index,
    #     reconstruction_loss,
    # )
    evaluation = evaluate_signal_mixing(
        num_neurons,
        num_behaviors,
        num_worms,
        num_top_contributors,
        contribution_rank,
        dataloader,
        model,
        attention_scheme
    )

    result_path = f'/store1/alicia/attention_predict/{experiment}'
    with open(f'{result_path}/evaluation_ckpt{model_ckpt}_k{num_top_contributors}.json', 'w') as json_file:
        json.dump(evaluation, json_file, indent=4)
    print(f'evaluation results written under {result_path}')

if __name__ == '__main__':
    main()
