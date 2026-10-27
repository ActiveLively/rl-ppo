from typing import Protocol

import torch


class VecEnv(Protocol):
    '''
    what PPO + Runner need from an env: batched over num_envs, torch tensors on `device`.
    step() auto-resets finished envs in the same step and returns the reset obs (gym SAME_STEP / Isaac Lab)
    '''
    num_envs: int
    obs_dim: int
    action_dim: int
    device: torch.device

    def get_obs(self) -> torch.Tensor: ... # [num_envs, obs_dim], raw (PPO normalizes)

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
        # obs [N, obs_dim], reward [N], terminated [N], truncated [N]

    def pop_metrics(self) -> dict[str, float]: ... # episode stats since the last call, for logging
