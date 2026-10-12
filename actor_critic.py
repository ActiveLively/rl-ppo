import numpy as np 
from mlp import MLP
import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int = 17):
        super().__init__()
        self.action_dim = action_dim 

        self.actor_hidden_dim = [256, 256]
        self.actor = MLP(
            input_dim = input_dim,
            output_dim = self.action_dim,
            hidden_dim = self.actor_hidden_dim,
            activation = 'tanh', 
            dropout = 0.0,
            norm = 'layer', 
            residual= False,
            init = "orthogonal",
            gain = (5/3),
            output_gain = 0.01
        )

        self.critic_hidden_dim = [256, 256]
        self.critic = MLP(
            input_dim = input_dim,
            output_dim = 1, # output value for the current state
            hidden_dim = self.critic_hidden_dim,
            activation = 'tanh', 
            dropout = 0.0,
            norm = 'layer', 
            residual= False,
            init = "orthogonal",
            gain = (5/3),
            output_gain = 1.0
        )
        #Actor Std Dev: state-independent learned parameter
        self.log_std = nn.Parameter (torch.zeros(self.action_dim))

    def act(self, x):
        mean = self.actor(x)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)

        action = dist.sample() # sample to explore actions
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(x)

        return action, log_prob, entropy, value

    def value(self, x):
        return self.critic(x).squeeze(-1)

    def evaluate(self, x, action):
        mean = self.actor(x)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(x)

        return log_prob, entropy, value
