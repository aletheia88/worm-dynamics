class Parameters():

    def __init__(
            self,
            n_layer,
            dropout,
            learning_rate,
            max_epochs,
            eval_epochs,
            batch_size,
            head_size,
            block_size,
            n_embd,
            ffwd_dim,
            device):

        self.n_head = int(n_embd / head_size)
        self.n_layer = n_layer
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.eval_epochs = eval_epochs
        self.batch_size = batch_size
        self.head_size = head_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.ffwd_dim = ffwd_dim
        self.device = device
