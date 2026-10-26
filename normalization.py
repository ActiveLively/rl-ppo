import torch


class RunningMeanStd():
    '''
    running mean/var on the GPU (parallel Welford update), torch version of gym's obs_rms.
    shape = () for scalars (e.g. value/return normalization), (obs_dim,) for observations
    '''
    def __init__(self, shape = (), device = "cpu", epsilon: float = 1e-4):
        self.shape = torch.Size(shape)
        self.mean = torch.zeros(self.shape, device=device)
        self.var = torch.ones(self.shape, device=device)
        self.count = epsilon

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, *self.shape) # [batch, *shape]
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta.square() * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    @property
    def std(self):
        return torch.sqrt(self.var + 1e-8)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state):
        self.mean, self.var, self.count = state["mean"], state["var"], state["count"]
