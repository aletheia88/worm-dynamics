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
    contribution_rank = {i: np.argsort(attn_matrix[i])[::-1] for i in range(N)}

    return contribution_rank


def benchmark_NfromB(
    num_top_contributors,
    contribution_rank,
    experiment,
    model_ckpt,
    model,
    num_worms,
    num_neurons,
    num_behaviors,
    dataloader,
):

    neuron_from_behaviors = {
        neuron_index: behaviors[:num_top_contributors]
        for neuron_index, behaviors in contribution_rank.items()
    }
    num_neurons = len(neuron_from_behaviors)

    ckpt_path = f'/store1/alicia/attention_predict/{experiment}/checkpoints/model_ckpt{model_ckpt}.pt'
    # load trained model
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    reconstruction_loss = torch.nn.MSELoss()

    left_slider = 0
    right_slider = 0
    max_length = 1600
    window_size = 400
    all_behavior_indices = list(range(num_neurons, num_neurons + num_behaviors))

    mse_per_neuron = {
        worm_index: {} for worm_index in range(num_worms)
    }
    mse_all_neurons = {
        neuron_index: {} for neuron_index in range(num_neurons)
    }
    # for each neuron, we compute mse_per_neuron
    # mse_per_neuron = {
    #     0: {
    #         'v1': mse_1,
    #         'v2': mse_2,
    #         ...,
    #         'v1-v2-..-vk': mse_k
    #     },
    #     ...
    #     n: {
    #         'v1': mse_1,
    #         'v2': mse_2,
    #         ...,
    #         'v1-v2-..-vk': mse_k
    #     }
    # }
    # then we recorgnize mse_per_neuron into the following
    # {
    #     'v1': [mse_11, mse_12, ..., mse_1n],
    #     'v2': [mse_11, mse_12, ..., mse_1n],
    #     ...,
    #     'v1-v2-..-vk': [mse_11, mse_12, ..., mse_1n],
    # }

    for i, (inputs, worm, start_frame, end_frame, ds) in tqdm(enumerate(dataloader)):

        print(f'\n=====worm: {worm.item()}=====\n')

        append = False
        worm_index = worm.item()

        if start_frame.item() == right_slider:
            append = True
            left_slider = right_slider
            right_slider = left_slider + window_size

        if start_frame.item() == max_length - window_size:
            append = True
            # reset the slider positions to append data from next worm
            left_slider = 0
            right_slider = 0

        if append:

            targets = deepcopy(inputs)
            recorded_neuron_indices = get_recorded_neuron_indices(targets, num_neurons)

            for neuron_index, behavior_indices in neuron_from_behaviors.items():

                # inputs: (num_samples=1, num_inputs, window_size)
                # behavior_indices = [index0, index1, index2, ...]
                # inputs contain [index0], [index0, index1], [index0, index1, index2]

                if neuron_index in recorded_neuron_indices:

                    contexts = establish_contexts(behavior_indices)
                    for context_id, context_indices in contexts.items():

                        # set all indices except context_indices to -10
                        ignore_indices = list(set(all_behavior_indices) -
                                              set(context_indices))
                        # TODO: different contexts give rise to the same loss
                        inputs[:, ignore_indices, :] = -1
                        outputs, _ = model(inputs)
                        loss = reconstruction_loss(
                            targets[:, neuron_index, :],
                            outputs[:, neuron_index, :]
                        )

                        mse_per_neuron[worm_index][context_id] = loss.item()

                    if neuron_index == 7 or neuron_index == 8:
                        print(f'mse of neuron {neuron_index}: {mse_per_neuron}')
                        # mse_all_neurons[neuron_index] = reformat(mse_per_neuron)


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


def reformat(mse_per_neuron):
    # mse_per_neuron -> target dictionary:
    # {
    #     'v1': [mse_11, mse_12, ..., mse_1n],
    #     'v2': [mse_11, mse_12, ..., mse_1n],
    #     ...,
    #     'v1-v2-..-vk': [mse_11, mse_12, ..., mse_1n],
    # }
    pass


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
    num_top_contributors = 3
    benchmark_NfromB(
        num_top_contributors,
        contribution_rank,
        experiment,
        model_ckpt,
        model,
        num_worms,
        num_neurons,
        num_behaviors,
        dataloader,
    )


if __name__ == '__main__':
    main()
