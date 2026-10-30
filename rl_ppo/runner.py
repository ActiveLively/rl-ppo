import os

import torch
from tqdm import tqdm

from .envs.base import VecEnv
from .logger import TrainingLogger
from .ppo import PPO


class Runner:
    '''
    owns the training loop (rollout -> update -> log) and checkpoint save/load.
    training scripts only build the env, networks and PPO, then call learn()
    '''
    def __init__(self, env: VecEnv, ppo: PPO, logger: TrainingLogger):
        self.env = env
        self.ppo = ppo
        self.logger = logger
        self.iteration = 0 # kept across learn() calls and checkpoints, so resumed runs log the right step

    def train(self, iters: int):
        steps_per_iter = self.ppo.steps * self.env.num_envs
        pbar = tqdm(range(iters), desc = "Training PPO")
        for _ in pbar:
            self.ppo.rollout()
            ppo_stats = self.ppo.update()
            self.iteration += 1

            metrics = self.logger.log_iteration(
                step = self.iteration * steps_per_iter,
                ppo_stats = ppo_stats,
                buffer = self.ppo.buffer,
                extra = {**self.env.pop_metrics(), **self.ppo.obs_stats()}
            )

            postfix = {"avg_reward": f"{metrics['rollout/reward_mean']:.3f}", "kl": f"{ppo_stats['kl']:.4f}"}
            if "rollout/success_rate" in metrics:
                postfix["success"] = f"{metrics['rollout/success_rate']:.2f}"
            pbar.set_postfix(postfix)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok = True)
        ckpt = {
            "model": self.ppo.actor_critic.state_dict(),
            "optimizer": self.ppo.optimizer.state_dict(),
            "iteration": self.iteration,
        }
        if self.ppo.obs_rms is not None:
            # numpy so visualize.py can load them straight into gym's NormalizeObservation
            ckpt["obs_mean"] = self.ppo.obs_rms.mean.cpu().numpy()
            ckpt["obs_var"] = self.ppo.obs_rms.var.cpu().numpy()
            ckpt["obs_count"] = self.ppo.obs_rms.count
            ckpt["obs_clip"] = self.ppo.obs_rms.clip
        if self.ppo.value_rms is not None:
            ckpt["value_rms"] = self.ppo.value_rms.state_dict()
        torch.save(ckpt, path)

    def load(self, path):
        '''resume from a checkpoint written by save()'''
        device = self.env.device
        ckpt = torch.load(path, map_location = device, weights_only = False)
        self.ppo.actor_critic.load_state_dict(ckpt["model"])
        self.ppo.optimizer.load_state_dict(ckpt["optimizer"])
        self.iteration = ckpt["iteration"]
        if self.ppo.obs_rms is not None:
            self.ppo.obs_rms.load_state_dict({
                "mean": torch.as_tensor(ckpt["obs_mean"], device = device),
                "var": torch.as_tensor(ckpt["obs_var"], device = device),
                "count": ckpt["obs_count"],
            })
        if self.ppo.value_rms is not None:
            self.ppo.value_rms.load_state_dict(ckpt["value_rms"])
