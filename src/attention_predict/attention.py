import torch


class Head(torch.nn.Module):

    def __init__(self, attention_mask, embedding_dims, num_inputs, device):

        super().__init__()
        self.sqrt_dk = num_inputs**0.5
        self.attention_mask = attention_mask

        self.key_weights = torch.nn.Linear(
                num_inputs,
                num_inputs,
                bias=False,
                device=device)
        self.query_weights = torch.nn.Linear(
                num_inputs,
                num_inputs,
                bias=False,
                device=device)
        self.value_weights = torch.nn.Linear(
                embedding_dims,
                embedding_dims,
                bias=False,
                device=device)
        # self.register_buffer('mask', attention_mask)

    def forward(self, key, query, value):

        weighted_key = self.key_weights(key)
        weighted_query = self.query_weights(query)
        weighted_value = self.value_weights(value)
        # attention_matrix shape: (num_samples, num_inputs, num_inputs)
        attention_matrix = weighted_query @ weighted_key.transpose(-2, -1) / self.sqrt_dk

        # apply attention masking
        attention_matrix = attention_matrix.masked_fill(self.attention_mask,
                                                        float('-inf'))
        attention_matrix = torch.nn.functional.softmax(attention_matrix, dim=-1)
        attention_matrix = torch.where(
                torch.isnan(attention_matrix),
                torch.zeros_like(attention_matrix),
                attention_matrix)
        attention_outputs = attention_matrix @ weighted_value

        return attention_outputs, attention_matrix


class MultiHeadAttention(torch.nn.Module):

    def __init__(
        self,
        attention_mask,
        embedding_dims,
        num_inputs,
        num_heads,
        device
    ):
        super().__init__()
        # single-headed self-attention
        self.head = Head(
                attention_mask,
                embedding_dims,
                num_inputs,
                device)
        self.key = torch.eye(num_inputs, device=device)
        self.query = torch.eye(num_inputs, device=device)
        # multi-headed self-attention (just an idea)
        # self.heads = torch.nn.ModuleList([Head(
        #         attention_mask,
        #         embedding_dims,
        #         num_inputs,
        #         device
        #     ) for _ in range(num_heads)])

    def forward(self, inputs):
        # single-headed self-attention
        num_samples = inputs.shape[0]
        key = self.key.unsqueeze(0).expand(num_samples, -1, -1)
        query = self.query.unsqueeze(0).expand(num_samples, -1, -1)

        return self.head(key=key, query=query, value=inputs)
        # multi-headed self-attention
        # return torch.cat([head(
        #         key=inputs,
        #         query=inputs,
        #         value=inputs
        #     )[0] for head in self.heads])


class AttentionBlock(torch.nn.Module):

    def __init__(
        self,
        embedding_dims,
        num_neurons,
        num_behaviors,
        attention_scheme,
        device
    ):
        super().__init__()

        ### fully attended attention matrix
        # -    |-----N-----|--B--|
        # |    |xxxxx..xxxx|x...x|
        # |    |xxxxx..xxxx|x...x|
        # N    |xxxxx..xxxx|x...x|
        # |    |xxxxx..xxxx|x...x|
        # |    |xxxxx..xxxx|x...x|
        # -    |...........|.....|
        # |    |xxxxx..xxxx|x...x|
        # B    |xxxxx..xxxx|x...x|
        # |    |-----------|-----|
        ### 'NfromN': attending neurons to reconstruct neurons
        # -    |-----N-----|--B--|
        # |    |0..xxxxxxxx|0...0|
        # |    |..0..xxxxxx|0...0|
        # N    |xx..0..xxxx|0...0|
        # |    |xxxx..0..xx|0...0|
        # |    |xxxxxx..0..|0...0|
        # -    |-----------|-----|
        # |    |00000..0000|0...0|
        # B    |00000..0000|0...0|
        # |    |-----------|-----|
        ### 'BfromN': attending neurons to reconstruct behaviors
        # -    |-----N-----|--B--|
        # |    |0.........0|0...0|
        # |    |0.0.......0|0...0|
        # N    |0..0......0|0...0|
        # |    |0.....0...0|0...0|
        # |    |0.......0.0|0...0|
        # -    |-----------|-----|
        # |    |xxxxx..xxxx|0...0|
        # B    |xxxxx..xxxx|0...0|
        # |    |-----------|-----|

        N = num_neurons
        B = num_behaviors
        num_inputs = N + B

        attention_quadrants = {
            'nn': torch.eye(N, device=device),
            # 'bb': torch.eye(B, B, device=device),
            'bb': torch.zeros(B, B, device=device),
            # use BfromB as control experiment
            # expect the network learns identity transform
            'nb': torch.zeros(N, B, device=device),
            'bn': torch.zeros(B, N, device=device),
        }
        inattention_quadrants = {
            'nn': torch.ones(N, N, device=device),
            'bb': torch.ones(B, B, device=device),
            'nb': torch.ones(N, B, device=device),
            'bn': torch.ones(B, N, device=device),
        }
        attention_mask = torch.zeros((num_inputs, num_inputs),
                                     dtype=torch.bool, device=device)

        if attention_scheme == 'NfromN':
            # top-left attention matrix
            attention_mask[:N, :N] = attention_quadrants['nn']
            # top-right attention matrix
            attention_mask[:N, N:] = inattention_quadrants['nb']
            # bottom-left attention matrix
            attention_mask[N:, :N] = inattention_quadrants['bn']
            # bottom-right attention matrix
            attention_mask[N:, N:] = inattention_quadrants['bb']

        elif attention_scheme == 'NfromB':
            attention_mask[:N, :N] = inattention_quadrants['nn']
            attention_mask[:N, N:] = attention_quadrants['nb']
            attention_mask[N:, :N] = inattention_quadrants['bn']
            attention_mask[N:, N:] = inattention_quadrants['bb']

        elif attention_scheme == 'BfromB':
            attention_mask[:N, :N] = inattention_quadrants['nn']
            attention_mask[:N, N:] = inattention_quadrants['nb']
            attention_mask[N:, :N] = inattention_quadrants['bn']
            attention_mask[N:, N:] = attention_quadrants['bb']

        elif attention_scheme == 'BfromN':
            attention_mask[:N, :N] = inattention_quadrants['nn']
            attention_mask[:N, N:] = inattention_quadrants['nb']
            attention_mask[N:, :N] = attention_quadrants['bn']
            attention_mask[N:, N:] = inattention_quadrants['bb']

        elif attention_scheme == 'all':
            attention_mask[:N, :N] = attention_quadrants['nn']
            attention_mask[:N, N:] = attention_quadrants['nb']
            attention_mask[N:, :N] = attention_quadrants['bn']
            attention_mask[N:, N:] = attention_quadrants['bb']

        self.attention_mask = attention_mask
        # self.attention = torch.nn.MultiheadAttention(
        #         embedding_dims,
        #         kdim=embedding_dims,
        #         vdim=embedding_dims,
        #         num_heads=1,
        #         batch_first=True,
        #         device=device)
        self.attention = MultiHeadAttention(
                attention_mask,
                embedding_dims,
                num_inputs,
                num_heads=1,
                device=device)

    def forward(self, inputs):

        # attention_outputs, attention_weights = self.attention(
        #         inputs,
        #         inputs,
        #         inputs,
        #         attn_mask=self.attention_mask)
        attention_outputs, attention_weights = self.attention(inputs)

        return attention_outputs, attention_weights
