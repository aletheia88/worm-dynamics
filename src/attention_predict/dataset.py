import torch
import math
import numpy as np


class SimpleSumDataset(torch.utils.data.IterableDataset):

    def __init__(self, input_dims):

        self.num_inputs = 3
        self.input_dims = input_dims

    def __iter__(self):
        torch.seed()
        while True:
            yield self.create_sample()

    def create_sample(self):

        a = torch.rand((self.input_dims,))
        b = torch.rand((self.input_dims,))
        c = a + b

        return torch.vstack((a, b, c))


class SimpleNonlinearDataset(torch.utils.data.IterableDataset):

    def __init__(self, input_dims):

        self.num_inputs = 4
        self.input_dims = input_dims

    def __iter__(self):
        torch.seed()
        while True:
            yield self.create_sample()

    def create_sample(self):

        a = torch.rand((self.input_dims,))
        b = 1.0 - a
        c = torch.rand((self.input_dims,))
        d = torch.cos(c * 2 * math.pi)

        return torch.vstack((a, b, c, d))


class CElegansDataset(torch.utils.data.Dataset):

    def __init__(self, path, window_size, device, slices=None):

        # data: (m, n, l)
        # m is number of worms
        # n is number of variables
        # l is length of sequence
        self.data = np.load(path).astype("float32")
        if slices:
            self.data = self.data[:, :, slices]
        self.window_size = window_size
        self.num_worms, self.num_variables, self.num_frames = self.data.shape
        self.device = device

    def __len__(self):
        return self.num_worms * (self.num_frames - self.window_size)

    def __getitem__(self, index):

        worm = index // (self.num_frames - self.window_size)
        start_frame = index % (self.num_frames - self.window_size)
        end_frame = start_frame + self.window_size

        # return: (n, window_size)
        return torch.tensor(
                self.data[worm, :, start_frame:end_frame],
                device=self.device,
                dtype=torch.float32
            ), worm, start_frame, end_frame - 1


class CElegansDatasetPlus(torch.utils.data.Dataset):

    def __init__(
            self,
            data_path,
            dataset_path,
            window_stride,
            window_size,
            device,
            slices=None
        ):

        self.data = np.load(data_path).astype("float32")
        self.datasets = np.load(dataset_path)

        if slices:
            self.data = self.data[:, :, slices]
        self.window_stride = window_stride
        self.window_size = window_size
        self.num_worms, self.num_variables, self.num_frames = self.data.shape
        self.device = device

    def __len__(self):

        num_windows = (self.num_frames - self.window_size) // self.window_stride + 1
        return self.num_worms * num_windows

    def __getitem__(self, index):

        num_windows_per_worm = (self.num_frames - self.window_size) // \
            self.window_stride + 1
        worm = index // num_windows_per_worm
        window_index = index % num_windows_per_worm

        start_frame = window_index * self.window_stride
        end_frame = start_frame + self.window_size
        # return: (n, window_size)
        return torch.tensor(
                self.data[worm, :, start_frame:end_frame],
                device=self.device,
                dtype=torch.float32
            ), worm, start_frame, end_frame - 1, self.datasets[worm]


if __name__ == "__main__":

    device = "cuda:2"
    window_stride = 400
    window_size = 400
    batch_size = 1
    base = '/home/alicia/notebook/alicia/worm-dynamics'
    ds_name = 'AVA_MC_SMDV_test'
    dataset = CElegansDatasetPlus(
        f'{base}/data/{ds_name}.npy',
        f'{base}/data/{ds_name}_ds.npy',
        window_stride=window_stride,
        window_size=window_size,
        device=device,
        slices=slice(0, 1600)
    )
    print(dataset.datasets)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )
    mse_per_dataset = {ds: {col: [] for col in range(6)} for ds in dataset.datasets}
    print(mse_per_dataset)
    # for n_iter, (inputs, worm, start_frame, end_frame, ds) in enumerate(dataloader):
    #     print(f'Iteration: {n_iter}')
    #     print(f'inputs: {inputs.shape}')
    #     print(f'dataset name: {ds[0]}')
    #     print(f'worm: {worm.item()} (start, end): {(start_frame.item(), end_frame.item())}')
