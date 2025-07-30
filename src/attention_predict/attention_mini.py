import torch


class Head(torch.nn.Module):

    def __init__(self, attention_mask, embedding_dims, attention_dims, device):

        super().__init__()

        self.sqrt_dk = attention_dims['k'][1]**0.5
        self.attention_mask = attention_mask

        self.key_weights = torch.nn.ModuleList([
            torch.nn.Linear(
                *attention_dims['w_k'],  # (N, N)
                bias=False
            ),
            torch.nn.Linear(
                *attention_dims['k'],  # (B, N)
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

        for i, linear in enumerate(self.query_weights):
            query = linear(query)
        weighted_query = query

        for i, linear in enumerate(self.key_weights):
            key = linear(key)
        weighted_key = key  # keys shape: e.g., (N, B) for BfromN

        # attention_matrix shapes:
        # (num_samples, B, N), (_, N, B), (_ , N, N), (_, B, B)
        attention_matrix = (weighted_query @ weighted_key).transpose(1, 2) \
            / self.sqrt_dk
        # apply attention masking
        attention_matrix = attention_matrix.masked_fill(self.attention_mask,
                                                        float('-inf'))
        attention_matrix = torch.nn.functional.softmax(
                attention_matrix, dim=-1)
        attention_outputs = attention_matrix @ weighted_value

        return attention_outputs, attention_matrix


class MultiHeadAttention(torch.nn.Module):

    def __init__(
        self,
        attention_mask,
        embedding_dims,
        attention_dims,  # dimensions of K, Q, and their weights
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

        return self.head(key, query, value)


class AttentionBlockMini(torch.nn.Module):

    def __init__(self, embedding_dims, N, B, attention_scheme, device):

        super().__init__()

        path = '/home/alicia/notebook/alicia/worm-dynamics/src/attention_predict'
        if attention_scheme == 'connectome':
            connectivity_matrix = torch.tensor(
                torch.load(f'{path}/connectome/sum_connectivity.pt'),
                dtype=torch.bool,
                device=device)
        elif attention_scheme == 'anticonnectome':
            connectivity_matrix = torch.tensor(
                torch.load(f'{path}/connectome/anti_connectivity.pt'),
                dtype=torch.bool,
                device=device)
        elif attention_scheme == 'randconnectome':
            connectivity_matrix = torch.tensor(
                torch.load(f'{path}/connectome/random_connectivity.pt'),
                dtype=torch.bool,
                device=device)
        else:
            connectivity_matrix = None

        attention_quadrants = {
            'nn': torch.eye(N, dtype=torch.bool, device=device),
            'bb': torch.zeros((B, B), dtype=torch.bool, device=device),
            'nb': torch.zeros((N, B), dtype=torch.bool, device=device),
            'bn': torch.zeros((B, N), dtype=torch.bool, device=device),
            'cn': connectivity_matrix,
            '1n': torch.zeros((1, N-1), dtype=torch.bool, device=device)
        }
        if attention_scheme == 'NfromN':
            attention_mask = attention_quadrants['nn']
            attention_dims = {
                'k': (N, N), 'q': (N, N),
                'w_k': (N, N), 'w_q': (N, N)
            }
        if attention_scheme == 'NfromB':
            attention_mask = attention_quadrants['nb']
            attention_dims = {
                'k': (B, N), 'q': (B, B),
                'w_k': (B, B), 'w_q': (B, B)
            }
        if attention_scheme == 'BfromN':
            attention_mask = attention_quadrants['bn']
            attention_dims = {
                'k': (N, B), 'q': (N, N),
                'w_k': (N, N), 'w_q': (N, N)
            }
        if attention_scheme == 'BfromB':
            attention_mask = attention_quadrants['bb']
            attention_dims = {
                'k': (B, B), 'q': (B, B),
                'w_k': (B, B), 'w_q': (B, B)
            }
        if attention_scheme in [
            'connectome',
            'anticonnectome',
            'randconnectome'
        ]:
            attention_mask = attention_quadrants['cn']
            attention_dims = {
                'k': (N, N), 'q': (N, N),
                'w_k': (N, N), 'w_q': (N, N)
            }
        if attention_scheme == '1fromN':
            attention_mask = attention_quadrants['1n']
            attention_dims = {
                'k': (N-1, 1), 'q': (N-1, N-1),
                'w_k': (N-1, N-1), 'w_q': (N-1, N-1)
            }

        self.attention = MultiHeadAttention(
            attention_mask,
            embedding_dims,
            attention_dims,
            num_heads=1,
            device=device
        )

    def forward(self, inputs):

        attention_outputs, attention_weights = self.attention(inputs)

        return attention_outputs, attention_weights


class AttentionBlockFixed(torch.nn.Module):

    """
    Fix uniform attention scores for attending to all context variables, except
    disabling attention to itself.
    """

    def __init__(
        self,
        embedding_dims,
        N,
        B,
        attention_scheme,
        perturb_index,
        device,
    ):
        super().__init__()

        if attention_scheme == 'NfromN':
            self.attention = (1 - torch.eye(N, N)).to(device)
            # self.attention = torch.ones((N, N), device=device)
        if attention_scheme == 'NfromB':
            self.attention = torch.ones((N, B), device=device)
        if attention_scheme == 'BfromN':
            self.attention = torch.ones((B, N), device=device)
        if attention_scheme == 'perturb' and perturb_index != None:
            self.attention = torch.zeros((N, N), device=device)
            self.attention[:, perturb_index] = 1

    def forward(self, embeddings):

        num_samples = embeddings.shape[0]
        # embeddings: (num_samples, num_context_variables, embeddings_dims)
        attention_matrix = self.attention.unsqueeze(0).expand(num_samples, -1, -1)

        return attention_matrix @ embeddings
