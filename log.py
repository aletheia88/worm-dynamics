
class Log:
    def __init__(self, attention_log_freq, loss_log_freq):
        self.attention_scores = []
        self.mse_losses = []
        self.iterations = []
        self.attention_log_freq = attention_log_freq
        self.loss_log_freq = loss_log_freq

    def log_iteration(self, num_iteration):
        self.iterations.append(num_iteration)

    def log_loss(self, loss):

        if self.iterations[-1] % self.loss_log_freq == 0:
            self.mse_losses.append(loss)

    def log_attention(self, attention_score):

        if self.iterations[-1] % self.attention_log_freq == 0:
            self.attention_scores.append(attention_score)

