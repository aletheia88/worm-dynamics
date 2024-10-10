from attention_predict.dataset import CElegansDatasetPlus
from attention_predict.unet import UNet
from copy import deepcopy
import numpy as np
import torch


def evaluate(
        model,
        model_ckpt_path,
        mask_indices,
        data_path, # testing datasets
        dataset_path,
        num_inputs,
        num_neurons,
        num_samples=100,
        max_length=1600,
        window_size=400,
        window_stride=1,
        device='cpu'):

    reconstruction_loss = torch.nn.MSELoss()

    # Load the trained model
    checkpoint = torch.load(model_ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    dataset = CElegansDatasetPlus(
            data_path,
            dataset_path,
            window_stride=window_stride,
            window_size=window_size,
            device=device,
            slices=slice(0, max_length))
    print(f'dataset length: {len(dataset)}')
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            drop_last=False) # shuffle to get random windows per worm

    # mse_per_dataset = {
    # '2022-06-14-13': {
    #   0: [mse1, mse2, ..., mse100] # column 0 reconstruction errors
    #   1: [mse1, mse2, ..., mse100] # column 1 reconstruction errors
    #   2: [mse1, mse2, ..., mse100]} # column 2 reconstruction errors
    # }
    # find datasets that have ground truth for all columns
    datasets_with_gt = get_datasets_with_all_ground_truth(data_path, dataset_path,
                                                          num_neurons)
    mse_per_dataset = {ds: {col: [] for col in range(num_inputs)}
                       for ds in datasets_with_gt}
    sample_metainfo = {ds: [] for ds in datasets_with_gt}

    for i, (inputs, worm, start_frame, end_frame, dataset) in enumerate(dataloader):

        ds = dataset[0]

        if ds not in datasets_with_gt:
            continue

        samples_filled = True
        for ds in datasets_with_gt:
            if len(mse_per_dataset[ds][0]) < num_samples:
                samples_filled = False
                break

        if samples_filled:
            return mse_per_dataset, sample_metainfo

        targets = deepcopy(inputs)
        inputs[:, mask_indices, :] = 0
        outputs = model(inputs)

        sample_metainfo[ds].append((start_frame.item(), end_frame.item()))
        for i in range(num_inputs):
            mse_per_dataset[ds][i].append(
                    reconstruction_loss(targets[0, i, :], outputs[0, i, :]).item())


def get_datasets_with_all_ground_truth(data_path, dataset_path, num_neurons):

    data = np.load(data_path)
    datasets = np.load(dataset_path)
    filtered_datasets = []

    for i, dataset in enumerate(datasets):

        ground_truth_available = True
        for j in range(num_neurons):
            if (np.max(data[i, j, :]) == 0 and np.min(data[i, j, :]) == 0):
                ground_truth_available = False
                break

        if ground_truth_available:
            filtered_datasets.append(dataset)

    return filtered_datasets


if __name__ == "__main__":

    device = 'cuda:2'

    ds_name = 'AVA_MC_SMDV'
    num_neurons = 3
    depth = 5
    base = '/home/alicia/notebook/alicia/worm-dynamics'
    in_channels = np.load(f'{base}/data/{ds_name}.npy').shape[1]
    out_channels = in_channels
    num_inputs = in_channels
    model = UNet(depth, in_channels, out_channels, unet_dim=1).to(device)
    exp = 'exp_2024100501'
    ckpt_dir = f'/home/alicia/store1/alicia/attention_predict/{exp}/checkpoints'
    model_ckpt_path = f'{ckpt_dir}/model_ckpt100.pt'

    # AVA | MC | SMDV | velocity | pumping | head angle
    mask_indices = [0, 2]
    data_path = f'{base}/data/{ds_name}_test.npy'
    dataset_path = f'{base}/data/{ds_name}_test_ds.npy'

    mse_per_dataset, sample_metainfo = evaluate(
        model,
        model_ckpt_path,
        mask_indices,
        data_path,
        dataset_path,
        num_inputs,
        num_neurons,
        device=device)
    for ds, mse_loss in mse_per_dataset.items():
        print(f'{ds}: {len(mse_loss[0])}')
    print('Number of unique frame chunks')
    for ds, frame_chunk in sample_metainfo.items():
        unique_frame_chunk = np.unique(np.array(frame_chunk), axis=1)
        print(f'{ds}: {unique_frame_chunk.shape[0]}')
