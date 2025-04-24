from attention_predict.attention_model_mini import AttentionModelMini
from attention_predict.unet import ConvBlock
import torch


def test_conv_block_init():
    ConvBlock(6, 12, 3, padding="same", num_groups=6)


def test_attention_model_init():
    AttentionModelMini()


def test_attention_model_forward():
    model = AttentionModelMini()
    fake_input = torch.zeros(1, 4, 1600)
    model(fake_input)
