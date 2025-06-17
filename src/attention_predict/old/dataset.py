import torch
from torch.utils.data import Dataset


class CElegansDatasetPlus(Dataset):
    def __init__(
        self, data, datasets, window_stride, window_size, num_encoders, device
    ) -> None:
        self.data = data
        self.datasets = datasets
        self.num_encoders = num_encoders
        self.window_stride = window_stride
        self.window_size = window_size
        self.num_worms, self.num_variables, self.num_frames = self.data.shape
        self.device = device
        return

    def __len__(self):
        num_windows = (self.num_frames - self.window_size) // self.window_stride + 1
        return self.num_worms * num_windows

    def __getitem__(self, index):
        num_windows_per_worm = (
            self.num_frames - self.window_size
        ) // self.window_stride + 1
        worm = index // num_windows_per_worm
        window_index = index % num_windows_per_worm

        start_frame = window_index * self.window_stride
        end_frame = start_frame + self.window_size
        data_slice = torch.tensor(
            self.data[worm, :, start_frame:end_frame],
            device=self.device,
            dtype=torch.float32,
        )
        input = data_slice[: self.num_encoders, :]
        output = data_slice[self.num_encoders :, :]
        return input, output
