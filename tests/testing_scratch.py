import marimo

__generated_with = "0.13.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    from attention_predict.attention_model_mini import AttentionModelMini
    import torch.utils.benchmark as benchmark
    import attention_predict.alicia.attention_model_mini as alicia
    return AttentionModelMini, alicia, benchmark, torch


@app.cell
def _(torch):
    device = torch.device("cuda")
    num_encoders = 30
    num_decoders = 20 
    input_dims = (32, 30, 512)
    return device, input_dims, num_decoders, num_encoders


@app.cell
def _(AttentionModelMini, input_dims, num_decoders):
    model = AttentionModelMini(num_decoders, input_dims)
    print(model)
    return (model,)


@app.cell
def _(alicia, num_decoders, num_encoders):
    alicia_model = alicia.AttentionModelMini(
        4, # depth
        num_encoders, # num neurons
        num_decoders, # num behaviors
        "NfromB",
        512,
        device = 'cuda'
    )
    return (alicia_model,)


@app.cell
def _(alicia_model, device, model):
    model.to(device)
    model.compile(fullgraph=True, mode="max-autotune")
    alicia_model.to(device)
    return


@app.cell
def _(alicia_model, device, input_dims, model, torch):
    fake_input = torch.rand(*input_dims).to(device)
    model(fake_input)
    alicia_model(fake_input)
    return (fake_input,)


@app.cell
def _(alicia_model, benchmark, fake_input, model):
    t0 = benchmark.Timer(
        stmt='model(fake_input)',
        globals={'fake_input': fake_input, 'model': model})

    t1 = benchmark.Timer(
        stmt='alicia_model(fake_input)',
        globals={'fake_input': fake_input, 'alicia_model': alicia_model})

    print(t0.timeit(100))
    print(t1.timeit(100))
    return


if __name__ == "__main__":
    app.run()
