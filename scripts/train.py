from attention_predict.model import PredictModel
from attention_predict.dataset import SimpleSumDataset, SimpleNonlinearDataset, CElegansDataset
from tqdm import tqdm
import torch


def train(dataset, model, batch_size, num_iterations, num_epochs, learning_rate):

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False)

    reconstruction_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_average = 0

    for _ in range(num_epochs):
        for i, inputs in tqdm(enumerate(dataloader)):
            if i == num_iterations:
                break
            optimizer.zero_grad()
            outputs, weights = model(inputs)
            loss = reconstruction_loss(inputs, outputs)
            loss.backward()
            optimizer.step()

            loss_average += loss

            if i % 100 == 99:
                loss_average /= 100
                print(f"Iteration {i}, running average loss: {loss_average}")
                loss_average = 0

                print("Input:")
                print(inputs[0])
                print("Reconstruction:")
                print(outputs[0])
                print("Attention weights:")
                print(weights[0], flush=True)


if __name__ == "__main__":

    #### simple linear, small model, working

    # batch_size = 1
    # input_dims = 4
    # embedding_dims = 128

    # dataset = SimpleSumDataset(input_dims)
    # num_inputs = dataset.num_inputs
    # model = PredictModel(
        # num_inputs=num_inputs,
        # input_dims=input_dims,
        # embedding_dims=embedding_dims,
        # num_layers=1)

    #### simple nonlinear, big model, working (but slow)

    # batch_size = 32
    # input_dims = 4
    # embedding_dims = 1024

    # dataset = SimpleNonlinearDataset(input_dims)
    # num_inputs = dataset.num_inputs
    # model = PredictModel(
        # num_inputs=num_inputs,
        # input_dims=input_dims,
        # embedding_dims=embedding_dims,
        # num_layers=4)

    #### same, but with residual layers in MLPs and layer norm

    # batch_size = 32
    # input_dims = 4
    # embedding_dims = 1024

    # dataset = SimpleNonlinearDataset(input_dims)
    # num_inputs = dataset.num_inputs
    # model = PredictModel(
        # num_inputs=num_inputs,
        # input_dims=input_dims,
        # embedding_dims=embedding_dims,
        # num_layers=4,
        # residual=True,
        # normalize=True)

    #### test on real data
    device = "cuda:2"
    window_size = 100
    training_dataset = CElegansDataset(
        "data/celegans.npy",
        window_size=window_size,
        device=device,
        slices=slice(0, 1200)
    )

    batch_size = 32
    embedding_dims = 1024

    model = PredictModel(
        num_inputs=training_dataset.num_variables,
        input_dims=window_size,
        embedding_dims=embedding_dims,
        num_layers=4,
        residual=True,
        normalize=True,
        device=device
    ).to(device)

    train(
        training_dataset,
        model,
        batch_size,
        num_iterations=1_000_000,
        num_epochs=1000,
        learning_rate=1e-4
    )


