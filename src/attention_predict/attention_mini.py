import torch


class Head(torch.nn.Module):

    def __init__(self, attention_mask, embedding_dims, attention_dims, device):

        super().__init__()

        self.sqrt_dk = attention_dims['k'][1]**0.5
        self.attention_mask = attention_mask

        # BfromN
        self.key_weights = torch.nn.ModuleList([
            torch.nn.Linear(
                *attention_dims['w_k'], # (N, N)
                bias=False
            ),
            torch.nn.Linear(
                *attention_dims['k'], # (B, N)
                bias=False
            )]
        ).to(device)
        self.query_weights = torch.nn.ModuleList([
            torch.nn.Linear(
                *attention_dims['w_q'],
                bias=False
            ),
            torch.nn.Linear(
                *attention_dims['q'],
                bias=False,
            )]
        ).to(device)
        self.value_weights = torch.nn.Linear(
            embedding_dims,
            embedding_dims,
            bias=False,
            device=device
        )

    def forward(self, key, query, value):

        weighted_value = self.value_weights(value)
        print(f'weighted value: {weighted_value.shape}')

        for i, linear in enumerate(self.query_weights):
            query = linear(query)
            print(f'query {i}: {query.shape}')
        weighted_query = query
        print(f'weighted query: {weighted_query.shape}')

        for i, linear in enumerate(self.key_weights):
            key = linear(key)
            print(f'key {i}: {key.shape}')
        weighted_key = key # (N, B) for BfromN
        print(f'weighted key: {weighted_key.shape}')

        # attention_matrix shapes: (num_samples, B, N), (_, N, B), (_ , N, N), (_, B, B)
        attention_matrix = (weighted_query @ weighted_key).transpose(1, 2) / self.sqrt_dk
        print(f'attention matrix: {attention_matrix.shape}')
        # apply attention masking
        attention_matrix = attention_matrix.masked_fill(self.attention_mask,
                                                        float('-inf'))
        attention_matrix = torch.nn.functional.softmax(attention_matrix, dim=-1)
        attention_outputs = attention_matrix @ weighted_value

        return attention_outputs, attention_matrix


class MultiHeadAttention(torch.nn.Module):

    def __init__(
        self,
        attention_mask,
        embedding_dims,
        attention_dims, # dimensions of K, Q, and their weights
        num_heads,
        device
    ):
        super().__init__()

        self.head = Head(
            attention_mask,
            embedding_dims,
            attention_dims,
            device
        )
        self.key = torch.eye(attention_dims['w_k'][0], device=device)
        self.query = torch.eye(attention_dims['w_q'][0], device=device)

    def forward(self, value):

        num_samples = value.shape[0]
        key = self.key.unsqueeze(0).expand(num_samples, -1, -1)
        query = self.query.unsqueeze(0).expand(num_samples, -1, -1)
        print(f'key: {key.shape}')
        print(f'query: {query.shape}')
        print(f'value: {value.shape}')

        return self.head(key, query, value)


class AttentionBlockMini(torch.nn.Module):

    def __init__(self, embedding_dims, N, B, attention_scheme, device):

        super().__init__()

        attention_quadrants = {
            'nn': torch.eye(N, dtype=torch.bool, device=device),
            'bb': torch.zeros((B, B), dtype=torch.bool, device=device),
            'nb': torch.zeros((N, B), dtype=torch.bool, device=device),
            'bn': torch.zeros((B, N), dtype=torch.bool, device=device),
        }

        if attention_scheme == 'NfromN':
            attention_mask = attention_quadrants['nn']
            attention_dims = {'k': (N, N), 'q': (N, N), 'w_k': (N, N), 'w_q': (N, N)}

        elif attention_scheme == 'NfromB':
            attention_mask = attention_quadrants['nb']
            attention_dims = {'k': (N, B), 'q': (B, B), 'w_k': (B, B), 'w_q': (B, B)}

        elif attention_scheme == 'BfromN':
            attention_mask = attention_quadrants['bn']
            attention_dims = {'k': (N, B), 'q': (N, N), 'w_k': (N, N), 'w_q': (N, N)}

        elif attention_scheme == 'BfromB':
            attention_mask = attention_quadrants['bb']
            attention_dims = {'k': (B, B), 'q': (B, B), 'w_k': (B, B), 'w_q': (B, B)}

        self.attention = MultiHeadAttention(
            attention_mask,
            embedding_dims,
            attention_dims,
            num_heads=1,
            device=device
        )

    def forward(self, inputs):

        attention_outputs, attention_weights = self.attention(inputs)
        print(f'attention outputs: {attention_outputs.shape}')
        return attention_outputs, attention_weights
