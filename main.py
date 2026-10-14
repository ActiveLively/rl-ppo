import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm

from actor_critic import ActorCritic
from ppo import PPO
from buffer import Buffer


def run_ppo():

    # ppo parameters
    num_envs = 8
    timesteps = 200000
    steps = 256
    epochs = 12
    minibatch_size = 256
    lr = 3e-4
    clip_eps = 0.2
    c1_value = 0.5
    c2_entropy = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Buffer parameters
    # GAE gamma and lambda
    gamma = 0.99
    lam = 0.1

    actor_critic = ActorCritic(
        input_dim = env.obs_dim,
        actor_hidden_dim = [256,256],
        critic_hidden_dim = [256, 256],
        action_dim = 17,
        activation = "tanh",
        dropout = 0.0,
        norm = "layer",
        residual = False,
        init = "orthogonal",
        actor_gain = (5/3),
        critic_gain = (5/3),
        actor_output_gain = 0.01,
        critic_output_gain = 1.0,
        state_dependent = True
    ).to(device)

    buffer = Buffer(
        num_steps = steps, 
        num_envs = num_envs, 
        obs_dim = env.obs_dim, 
        action_dim = env.action_dim, 
        device = device, 
        gamma = gamma, 
        lam = lam
    )

    ppo = PPO(
        env = env,
        actor_critic= actor_critic,
        buffer = buffer,
        steps = steps,
        epochs = epochs,
        minibatch_size = minibatch_size,
        lr = lr,
        clip_eps = clip_eps,
        c1_value = c1_value,
        c2_entropy = c2_entropy
    )

    iters = timesteps // (steps * num_envs)
    pbar = tqdm(range(1, iters + 1), desc= "Training PPO")   
    for it in pbar:
        ppo.rollout()
        ppo.update()

