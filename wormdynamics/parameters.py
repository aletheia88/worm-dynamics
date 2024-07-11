from dataclasses import dataclass, field


@dataclass
class TransformerParameters:
    n_layer: int
    dropout: float
    learning_rate: float
    max_epochs: int
    eval_epochs: int
    batch_size: int
    head_size: int
    block_size: int
    n_embd: int
    ffwd_dim: int
    attention_span: int
    device: str
    n_head: int = field(init=False)

    def __post_init__(self):
        self.n_head = int(self.n_embd / self.head_size)

@dataclass
class UNetParameters:
    learning_rate: float
    max_epochs: int
    eval_epochs: int
    batch_size: int
    device: str

@dataclass
class DataParameters:
    noise_multiplier: float
    num_to_augment: int
