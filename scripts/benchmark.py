from attention_predict.attention_model import AttentionModel
from attention_predict.attention_model_2 import AttentionModel2
from attention_predict.attention_model_3 import AttentionModel3
from attention_predict.dataset import CElegansDatasetPlus
from attention_predict.reconstruct import reconstruct_traces, build_model, build_dataloader
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch

# get attention matrix and normalize according to the baseline
# for each row, rank the attention scores of each behavior from high to low
# get the top 5 behaviors (and their column indices) with the highest attention score
# delete each behavior from inputs and get model outputs
# compute MSE which contributes a dot on the final violin plot


def rank_contributors(reconstruction, attention_scheme, N):
    # N = num_neurons
    if attention_scheme == 'NfromB':
        attn_matrix = reconstruction[0]['attn_weights'][0][0, :N, N:]
    elif attention_scheme == 'BfromN':
        attn_matrix = reconstruction[0]['attn_weights'][0][0, N:, :N]
    elif attention_scheme == 'NfromN':
        attn_matrix = reconstruction[0]['attn_weights'][0][0, :N, :N]

    baseline_attn = 1 / attn_matrix.shape[1]
    normalized_attn_matrix = (attn_matrix - baseline_attn) / baseline_attn
    contribution_rank = {i: N + np.argsort(attn_matrix[i])[::-1] for i in range(N)}

    return contribution_rank


def load_model_weights(model, experiment, model_ckpt):

    model_ckpt_path = f'/store1/alicia/attention_predict/{experiment}/checkpoints/model_ckpt{model_ckpt}.pt'
    checkpoint = torch.load(model_ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return model


def evaluate_Ns():
    # for each neuron, we compute mse_per_neuron
    # mse_per_neuron = {
    #     0: {
    #         'v1': [mse_1, mse_1, mse_1, mse_1], # 4 windows therefore 4 mse_1
    #         'v2': [mse_2, mse_2, mse_2, mse_2],
    #         ...,
    #         'v1-v2-..-vk': [mse_k, mse_k, mse_k, mse_k]
    #     },
    #     ...
    #     n: {
    #         'v1': [mse_1, mse_1, mse_1, mse_1]
    #         'v2': [mse_2, mse_2, mse_2, mse_2]
    #         ...,
    #         'v1-v2-..-vk': [mse_k, mse_k, mse_k, mse_k]
    #     }
    # }
    # then we recorgnize mse_per_neuron into the following
    # {
    #     'v1': [total_mse_11, total_mse_12, ..., total_mse_1n],
    #     'v2': [total_mse_11, total_mse_12, ..., total_mse_1n],
    #     ...,
    #     'v1-v2-..-vk': [total_mse_11, total_mse_12, ..., total_mse_1n],
    # }

    # reconstruction_loss = torch.nn.MSELoss()
    # all_indices = list(range(num_neurons + num_behaviors))
    # dataset = dataloader.dataset
    # num_windows = max_length // window_size
    pass


@torch.no_grad()
def evaluate_N_from_Bs(
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


def main():

    architecture = 'attention_model_2'
    attention_scheme = 'NfromB'
    ds_name = 'steve1230_norm_eval'
    device = 'cuda:2'
    model_ckpt = 1570
    experiment = 'exp_2024122401'
    num_neurons = 14
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
        num_neurons
    )
    model = load_model_weights(model, experiment, model_ckpt)
    num_top_contributors = 3
    num_windows = 4
    dataset = dataloader.dataset
    all_indices = list(range(num_neurons + num_behaviors))
    worm_index = 17
    neuron_index = 7
    reconstruction_loss = torch.nn.MSELoss()

    reconstruction_mse = evaluate_N_from_Bs(
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
    )
    print(reconstruction_mse)

if __name__ == '__main__':
    main()
