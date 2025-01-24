import attention_predict
import torch


if __name__ == '__main__':

    depth = 3
    num_neurons = 3
    num_behaviors = 3
    num_inputs = num_neurons + num_behaviors
    window_size = 400
    batch_size = 2
    device = 'cuda:3'
    # (batch, channels, height, width)
    x = torch.rand(batch_size, num_inputs, window_size).to(device)
    model = attention_predict.AttentionModel2(
            depth,
            num_neurons,
            num_behaviors,
            window_size,
            attention_scheme='NfromB',
            device=device)
    y, level_attn_weights = model(x)
    print(f'outputs dim: {y.shape}')
    for attn_weights in level_attn_weights:
        print(f'weights dim: {attn_weights.shape}')
        print(attn_weights)
