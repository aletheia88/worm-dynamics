from torch import Tensor
from torch.utils.data import Dataset


class SuffledWormData(Dataset):
    data: Tensor
    window_stride: int
    window_size: int
    num_windows: int
    length: int

    def __init__(self, data: Tensor, window_stride: int, window_size: int) -> None:
        self.data = data
        num_worms, num_variables, num_frames = data.shape
        self.window_stride = window_stride
        self.window_size = window_size
        self.num_windows = (num_frames - self.window_size) // self.window_stride + 1
        self.length = num_worms * self.num_windows

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tensor:
        worm = index // self.num_windows
        start_frame = (index % self.num_windows) * self.window_stride
        end_frame = start_frame + self.window_size
        return self.data[worm, :, start_frame:end_frame]
