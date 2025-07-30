from attention_predict.reconstruct_mini import (
        reconstruct_traces,
        build_mini_attention_model,
        build_dataloader
)
from pprint import pprint
from sklearn.model_selection import KFold
from tqdm import tqdm
import attention_predict
import json
import numpy as np
import torch


def build_dataloader_from_fold(ds_name, n_fold):

    random_seed = 2025
    device = 'cuda:3'
    k_folds = 5
    batch_size = 1
    base = '/home/alicia/store1/alicia/attention_predict'
    training_dataset = attention_predict.dataset.CElegansDatasetPlus(
        f'{base}/data/{ds_name}_train.npy',
        f'{base}/data/{ds_name}_train_ds.npy',
        window_stride=1,
        window_size=400,
        device=device,
        slices=slice(0, 1600)
    )
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)

    for fold, (training_ids, validation_ids) in enumerate(kfold.split(training_dataset)):

        if fold == n_fold:
            training_subset = torch.utils.data.Subset(training_dataset, training_ids)
            validation_subset = torch.utils.data.Subset(training_dataset, validation_ids)
            validation_dataloader = torch.utils.data.DataLoader(
                    validation_subset,
                    batch_size=batch_size,
                    shuffle=False
            )
            training_dataloader = torch.utils.data.DataLoader(
                    training_subset,
                    batch_size=batch_size,
                    shuffle=False
            )
            break

    return training_dataloader, validation_dataloader


def evaluate_on_fold(
    attention_scheme,
    ds_name,
    n_ckpt,
    experiment,
    depth,
    num_neurons,
    num_behaviors,
    num_fmaps,
    dataloader,
    target_variables,
    device,
):
    window_size = 400
    neuron_indices = list(range(num_neurons))
    behavior_indices = list(range(num_neurons, num_neurons+num_behaviors))
    ckpt_path = f'/store1/alicia/attention_predict/{experiment}/model_ckpt{n_ckpt}.pt'

    model = build_mini_attention_model(
        attention_scheme,
        depth,
        num_neurons,
        num_behaviors,
        window_size,
        num_fmaps,
        device
    )
    # load trained model
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    if attention_scheme == 'BfromN':
        input_indices = neuron_indices
        output_indices = behavior_indices

    if attention_scheme == 'NfromB':
        input_indices = behavior_indices
        output_indices = neuron_indices

    if attention_scheme in ['NfromN', 'connectome', 'anticonnectome']:
        input_indices = neuron_indices
        output_indices = neuron_indices

    if attention_scheme == 'BfromB':
        input_indices = behavior_indices
        output_indices = behavior_indices

    reconstruction_error = torch.nn.MSELoss(reduction='sum')
    mse_dict = {}
    max_samples = 1000000

    for i, (inputs, _, _, _, ds) in tqdm(enumerate(dataloader)):

        if i == max_samples:
            break

        outputs, _ = model(inputs[:, input_indices, :])
        targets = inputs[:, output_indices, :]
        ds_name = ds[0]

        if ds_name not in mse_dict.keys():
            mse_dict[ds_name] = {var: [] for var in target_variables.keys()}

        for var, j in target_variables.items():
            mse = reconstruction_error(targets[:, j, :], outputs[:, j, :])
            mse_dict[ds_name][var].append(mse.item())

    # average to get estimate of MSE per time point
    estimate_error = {}
    for ds_name, var_losses_dict in mse_dict.items():
        estimate_error[ds_name] = {}
        for var, losses in var_losses_dict.items():
            estimate_error[ds_name][var] = np.mean(losses) / window_size

    return estimate_error


def evaluate(
    attention_scheme,
    n_ckpt,
    experiment,
    depth,
    num_neurons,
    num_behaviors,
    num_fmaps,
    dataloader,
    target_variables,
    num_datasets,
    device,
    target_index,
    perturb_index,
):
    window_size = 400

    model = build_mini_attention_model(
        attention_scheme,
        depth,
        num_neurons,
        num_behaviors,
        window_size,
        num_fmaps,
        # TODO: add perturb_index
        perturb_index,
        device
    )
    print('model built!')
    reconstructed_traces = reconstruct_traces(
        model,
        attention_scheme,
        dataloader,
        n_ckpt,
        experiment,
        num_datasets,
        num_neurons,
        num_behaviors,
        device,
        target_index=target_index,
    )
    estimate_error = compute_mse_2ch(
        reconstructed_traces,
        target_variables,
        attention_scheme,
        slices=slice(5, 393)
    )
    return estimate_error


def compute_mse(reconstructed_traces, target_variables, attention_scheme):

    estimate_error = {}
    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.L1Loss()

    for _, output_dict in reconstructed_traces.items():

        # prediction, ground_truth: (num_output_variables, sequence_length)
        prediction = np.concatenate(
            output_dict['prediction'],
            axis=2).squeeze(0)
        ground_truth = np.concatenate(
            output_dict['ground_truth'],
            axis=2).squeeze(0)
        ds_name = output_dict['dataset']
        estimate_error[ds_name] = {}

        for var, i in target_variables.items():

            if np.max(ground_truth[i, :]) == -10 and \
                    np.min(ground_truth[i, :]) == -10:
                estimate_error[ds_name][var] = None
            else:
                estimate_error[ds_name][var] = criterion(
                    torch.tensor(prediction[i, :]),
                    torch.tensor(ground_truth[i, :])
                ).item()

    return estimate_error


def compute_mse_2ch(
    reconstructed_traces,
    target_variables,
    attention_scheme,
    slices=None,
):
    estimate_error = {}
    criterion = torch.nn.MSELoss()
    # slices = slice(1, 398)
    num_vecs = len(target_variables) * 2

    for _, output_dict in reconstructed_traces.items():

        num_windows = len(output_dict['prediction'])
        # output_dict['prediction']: [(1, 70, 400), ..., (1, 70, 400)]
        if slices == None:
            prediction = np.concatenate(output_dict['prediction'], axis=2).squeeze(0)
            ground_truth_activity = [
                np.zeros_like(output_dict['prediction'][k])
                for k in range(num_windows)
            ]
            for k in range(num_windows):
                for i, j in enumerate(range(0, num_vecs, 2)):
                    ground_truth_activity[k][:, i, :] = \
                            output_dict['ground_truth'][k][:, j, :][:]
        else:
            sliced_prediction = [
                predicted_window[:, :, slices]
                for predicted_window in output_dict['prediction']
            ]
            prediction = np.concatenate(sliced_prediction, axis=2).squeeze(0)
            ground_truth_activity = [
                np.zeros_like(sliced_prediction[k])
                for k in range(num_windows)
            ]
            for k in range(num_windows):
                for i, j in enumerate(range(0, num_vecs, 2)):
                    ground_truth_activity[k][:, i, :] = \
                            output_dict['ground_truth'][k][:, j, slices][:]

        ground_truth = np.concatenate(
            ground_truth_activity,
            axis=2
        ).squeeze(0)
        print(f'ground truth: {ground_truth.shape}')
        print(f'prediction: {prediction.shape}')

        ds_name = output_dict['dataset']
        estimate_error[ds_name] = {}

        for var, i in target_variables.items():

            if np.max(ground_truth[i, :]) == -10 and \
                    np.min(ground_truth[i, :]) == -10:
                estimate_error[ds_name][var] = None
            else:
                estimate_error[ds_name][var] = criterion(
                    torch.tensor(prediction[i, :]),
                    torch.tensor(ground_truth[i, :])
                ).item()

    return estimate_error


def main():

    ds_type = 'valid'
    device = 'cuda:1'

    # if ds_type == 'test':
    #     num_datasets = 20
    # elif ds_type == 'valid':
    #     num_datasets = 16

    # datasets: data0626, data0410, ...
    if ds_type == 'test':
        num_datasets = 10
    elif ds_type == 'valid':
        num_datasets = 20

    # datasets: data0715
    # if ds_type == 'test':
    #     num_datasets = 14
    # elif ds_type == 'valid':
    #     num_datasets = 27

    ds_name = f'data0626/data0626-00e_norm0505_{ds_type}'
    # ds_name = f'data0715e_norm0505_{ds_type}'
    attention_scheme = 'NfromN'
    depth = 5
    num_neurons = 70
    num_behaviors = 4
    num_fmaps = 64
    n_ckpt = 91
    fig4_neurons = [
            'RIB', 'RIC', 'RID', 'AUA', 'AVJ', 'AVK', 'AIM', 'AIY',
            'AVA', 'AVE', 'AIB', 'RIM', 'RIV', 'ADA', 'AVD', 'RIA',
            'AVH', 'AIN', 'AIZ', 'URB', 'ALA', 'RMG', 'RMD', 'RMDD',
            'RMDV', 'RME', 'RMEV', 'RMED', 'SAAV', 'SMDV', 'IL1L',
            'IL1R', 'IL1D', 'IL1V', 'URYD', 'URYV', 'BAG', 'ASG',
            'CEPD', 'CEPV', 'OLL', 'OLQD', 'OLQV', 'IL2L', 'IL2R',
            'IL2D', 'IL2V', 'URAD', 'URAV', 'ADE', 'FLP', 'AQR',
            'URX', 'ADL', 'ASH', 'ASEL', 'ASER', 'AWA', 'AWB',
            'AWC', 'I1', 'I2', 'I3', 'NSM', 'M1', 'M3', 'M4',
            'M5', 'MC', 'MI']
    target_variables = {
        neuron: i for i, neuron in enumerate(fig4_neurons)
    }
    # target_variables = {'AIN': 0}

    # noise = '0.5'
    # tag = '2ch_s20_fixedattn_corr_groups2'
    # tag = '2ch_s20_fixedattn-hard_split0'
    tag = 'currbest_fix_shuffle'
    experiment = f'whole_brain/depth5_nfmaps64_lr5e-05_{attention_scheme}_norm0505_{tag}'

    valid_dataloader = build_dataloader(ds_name, device)
    print('dataloder built!')
    # valid_data_segments = {}
    # valid_datasets = []

    # for n_iter, (inputs, ds_index, t_start, t_end, ds_name) in
    # tqdm(enumerate(valid_dataloader)):
    #     if ds_name[0] not in valid_datasets:
    #         valid_datasets.append(ds_name[0])
    #         valid_data_segments[ds_name[0]] = [t_start.item()]
    #     else:
    #         valid_data_segments[ds_name[0]].append(t_start.item())

    # print(len(valid_datasets), valid_datasets)
    # pprint(valid_data_segments)
    # estimate_error = evaluate_on_fold(
    #     attention_scheme,
    #     ds_name,
    #     n_ckpt,
    #     experiment,
    #     depth,
    #     num_neurons,
    #     num_behaviors,
    #     num_fmaps,
    #     valid_dataloader,
    #     target_variables,
    #     device)

    # target_index = 23
    # RMDD: 23
    # RMD: 22
    # AIN: 17
    target_index = None
    perturb_index = None
    estimate_error = evaluate(
        attention_scheme,
        n_ckpt,
        experiment,
        depth,
        num_neurons,
        num_behaviors,
        num_fmaps,
        valid_dataloader,
        target_variables,
        num_datasets,
        device,
        target_index,
        perturb_index,
    )
    # pprint(estimate_error)
    with open(f'eval/errorMSE_data0626e{ds_type}-shuffle_{attention_scheme}_{tag}.json', 'w') as f:
        json.dump(estimate_error, f, indent=4)


if __name__ == '__main__':
    main()
