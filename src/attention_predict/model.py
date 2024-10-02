import torch


class MlpLayer(torch.nn.Module):

    def __init__(
            self,
            dims_in,
            dims_out,
            activation=None,
            dropout=None,
            residual=False,
            normalize=False):

        super().__init__()

        self.residual = residual and dims_in == dims_out

        layers = [torch.nn.Linear(dims_in, dims_out)]

        if activation is not None:
            layers.append(activation())
        if dropout is not None:
            layers.append(torch.nn.Dropout(dropout))

        self.mlp = torch.nn.Sequential(*layers)

        if normalize:
            self.norm = torch.nn.LayerNorm(dims_out)
        else:
            self.norm = None

    def forward(self, x):

        y = self.mlp(x)
        if self.residual:
            y = y + x
        if self.norm is not None:
            y = self.norm(y)
        return y


class PredictModel(torch.nn.Module):

    def __init__(
            self,
            num_inputs,
            input_dims,
            embedding_dims,
            num_layers=3,
            dropout=None,
            residual=False,
            normalize=False,
            device="cuda:3"):

        super().__init__()
        self.num_inputs = num_inputs
        self.input_dims = input_dims
        self.embedding_dims = embedding_dims
        self.device = device
        # dimension = embedding + one-hot encoding
        self.query_dims = self.embedding_dims + num_inputs
        self.key_dims = self.embedding_dims + num_inputs
        self.value_dims = self.embedding_dims + num_inputs

        # one encoder per input
        self.encoders = []
        for i in range(num_inputs):
            pre_dims = input_dims
            next_dims = self.embedding_dims * 2
            activation = torch.nn.ReLU
            layers = []
            for l in range(num_layers):
                last_layer = l == num_layers - 1
                if last_layer:
                    next_dims = self.embedding_dims
                layers.append(
                    MlpLayer(
                        pre_dims,
                        next_dims,
                        activation,
                        None if last_layer else dropout,
                        residual,
                        False if last_layer else normalize
                    ).to(self.device)
                )
                pre_dims = next_dims

            encoder = torch.nn.Sequential(*layers)
            self.encoders.append(encoder)

        # single attention layer
        self.attention = torch.nn.MultiheadAttention(
            self.query_dims,
            kdim=self.key_dims,
            vdim=self.value_dims,
            num_heads=1,
            batch_first=True,
        )

        # one decoder per output (same number as inputs)
        self.decoders = []
        for i in range(num_inputs):
            pre_dims = self.value_dims
            next_dims = embedding_dims * 2
            activation = torch.nn.ReLU
            layers = []
            for l in range(num_layers):
                last_layer = l == num_layers - 1
                if last_layer:
                    next_dims = input_dims
                layers.append(
                    MlpLayer(
                        pre_dims,
                        next_dims,
                        None if last_layer else activation,
                        None if last_layer else dropout,
                        residual,
                        False if last_layer else normalize
                    ).to(self.device)
                )
                pre_dims = next_dims

            decoder = torch.nn.Sequential(*layers)
            self.decoders.append(decoder)

        self.one_hots = torch.eye(num_inputs, device=self.device)
        self.attn_mask = torch.eye(
                self.num_inputs,
                dtype=torch.bool,
                device=self.device)

    def forward(self, inputs):
        """

        Args:

            inputs: (tensor of shape ``(b, num_inputs, input_dims)``)
        """

        # number of samples in the batch
        num_samples = inputs.shape[0]

        keys = []
        for i in range(self.num_inputs):
            input_embeddings = self.encoders[i](inputs[:, i, :])
            one_hot_encoding = self.one_hots[i].repeat(num_samples, 1)
            key = torch.cat((input_embeddings, one_hot_encoding), 1)
            # key: (b, embedding_dims + num_inputs)
            key = torch.unsqueeze(key, axis=1)
            # key: (b, 1, embedding_dims + num_inputs)
            keys.append(key)

        keys = torch.cat(keys, axis=1)
        # keys: (b, num_inputs, embedding_dims + num_inputs)

        # prepare for "self-attention" where key = query = value
        queries = keys
        values = keys
        # attention with mask
        attn_output, attn_weights = self.attention(
            queries,
            keys,
            values,
            attn_mask=self.attn_mask)
        # attn_output: (b, num_inputs, embedding_dims + num_inputs)

        outputs = [
            self.decoders[i](attn_output[:, i, :])
            # output: (b, input_dims)
            for i in range(self.num_inputs)
        ]

        outputs = torch.stack(outputs, 1)
        # outputs: (b, num_inputs, input_dims)

        return outputs, attn_weights


if __name__ == "__main__":

    inputs = torch.rand(30, 4, 1600)
    num_inputs = 4
    input_dims = 1600
    embedding_dims = 1024
    model = PredictModel(
        num_inputs,
        input_dims,
        embedding_dims,
        residual=True)
    outputs, attn_weights = model(inputs)
