import attention_predict
import torch


if __name__ == '__main__':

    depth = 4
    num_neurons = 4
    num_behaviors = 3
    window_size = 400
    batch_size = 2
    attention_scheme = 'BfromN'
    device = 'cuda:3'
    num_fmaps = 16
    # (batch, channels, height, width)
    x = torch.rand(batch_size, num_neurons, window_size).to(device)
    model = attention_predict.AttentionModelMini(
            depth,
            num_neurons,
            num_behaviors,
            attention_scheme,
            window_size,
            num_fmaps=num_fmaps,
            device=device)
    y, level_attn_weights = model(x)
    print(f'outputs dim: {y.shape}')
    for attn_weights in level_attn_weights:
        print(f'attention weights: {attn_weights.shape}')
        print(attn_weights)
