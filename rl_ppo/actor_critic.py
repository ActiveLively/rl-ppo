import numpy as np 
from .mlp import MLP
import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 actor_hidden_dim: None, 
                 critic_hidden_dim: None,
                 action_dim: int = 17, 
                #  critic_output_dim: int = 1,
                 activation: str = "tanh",
                 dropout: float = 0.0,
                 norm: str = "layer",
                 residual: bool = False,
                 init: str = "orthogonal",
                 actor_gain: float = (5/3),
                 critic_gain: float = (5/3),
                 actor_output_gain: float = 0.01,
                 critic_output_gain: float = 1.0,
                 state_dependent: bool = False
                 ):
        super().__init__()

        # set the actor output dimensions depending on if we want a state-dependent std-dev
        self.state_dependent = state_dependent
        if state_dependent:
            actor_output_dim = action_dim * 2
        else:
            #Actor Std Dev: state-independent learned parameter
            actor_output_dim = action_dim
            self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.actor = MLP(
            input_dim = input_dim,
            hidden_dim = actor_hidden_dim,
            output_dim = actor_output_dim,
            activation = activation, 
            dropout = dropout,
            norm = norm, 
            residual= residual,
            init = init,
            gain = actor_gain,
            output_gain = actor_output_gain
        )

        self.critic = MLP(
            input_dim = input_dim,
            hidden_dim = critic_hidden_dim,
            output_dim = 1, # output value for the current state
            activation = activation, 
            dropout = dropout,
            norm = norm, 
            residual= residual,
            init = init,
            gain = critic_gain,
            output_gain = critic_output_gain
        )

    def get_mean_std(self, x):
        '''
        depends on state_dependence flag
        '''
        if self.state_dependent:
            mean, log_std = torch.chunk(self.actor(x), 2, dim = -1)
        else:
            mean = self.actor(x)
            log_std = self.log_std
        std = torch.exp(log_std.clamp(-5, 2))
        return mean, std

    def act(self, x):
        mean, std = self.get_mean_std(x)
        dist = Normal(mean, std)

        action = dist.sample() # sample to explore actions
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.value(x)

        return action, log_prob, value, mean, std # mean/std stored for the exact KL in update()

    def value(self, x):
        return self.critic(x).squeeze(-1)

    def evaluate(self, x, action):
        mean, std = self.get_mean_std(x)
        dist = Normal(mean, std)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(x)

        return log_prob, entropy, value, mean, std
