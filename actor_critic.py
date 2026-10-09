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
            activation = nn.ReLU(), 
            dropout = 0.0,
            norm = None, 
            residual= False 
        )
        self.critic_hidden_dim = [256, 256]
        self.critic = MLP(
            nput_dim = input_dim,
            output_dim = self.action_dim,
            hidden_dim = self.actor_hidden_dim,
            activation = nn.ReLU(), 
            dropout = 0.0,
            norm = None, 
            residual= False 
        )
        #Actor Std Dev: state-independent learned parameter
        self.log_std = nn.Parameter (torch.zeros(self.action_dim))



