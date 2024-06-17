class IterationLog():

    def __init__(
        self,
        iteration,
        attention,
        train_loss,
        valid_loss
    ):
        self.iteration = iteration
        self.attention = attention
        self.train_loss = train_loss
        self.valid_loss = valid_loss

class Log():

    def __init__(self, attention_log_freq):

        self.iteration_logs = []
        self.attention_log_freq = attention_log_freq

    def log_iteration(
        self,
        iteration,
        attention=None,
        train_loss=None,
        valid_loss=None
    ):
        self.iteration_logs.append(
            IterationLog(
                iteration,
                attention,
                train_loss,
                valid_loss
            )
        )
