import os
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Velocity-Flat-Anymal-D-v0")
parser.add_argument("--num_envs", type=int, default=4096)
AppLauncher.add_app_launcher_args(parser) # --headless, --device, ...
args = parser.parse_args()

# the sim app must be running before importing anything from isaaclab / isaaclab_tasks
simulation_app = AppLauncher(args).app

import torch
import gymnasium as gym
import isaaclab_tasks # registers the Isaac-* gym envs
from isaaclab_tasks.utils import parse_env_cfg
from tqdm import tqdm

from actor_critic import ActorCritic
from ppo import PPO
from buffer import Buffer
from logger import TrainingLogger
from isaaclab_env import IsaacLabEnv


def run_ppo():
    # PPO parameters (Isaac Lab rsl_rl defaults for locomotion)
    iters = 1500
    steps = 24
    epochs = 5
    num_minibatches = 4
    lr = 1e-3
    clip_eps = 0.2
    c1_value = 1.0
    c2_entropy = 0.005
    max_grad_norm = 1.0
    desired_kl = 0.01
    clip_value = True
    normalize_value = False
    normalize_obs = True
    gamma = 0.99
    lam = 0.95

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = IsaacLabEnv(gym.make(args.task, cfg=env_cfg), clip_actions=None) # e.g. 1.0 if the task has no action scaling
    device = torch.device(env.device)

    logger = TrainingLogger(
        log_dir = "runs",
        run_name = args.task,
        writers = ("jsonl",),
        config = dict(
            env_id = args.task, num_envs = env.num_envs, iters = iters, steps = steps, epochs = epochs,
            num_minibatches = num_minibatches, lr = lr, clip_eps = clip_eps, c1_value = c1_value,
            c2_entropy = c2_entropy, gamma = gamma, lam = lam, max_grad_norm = max_grad_norm,
            desired_kl = desired_kl, clip_value = clip_value, normalize_value = normalize_value,
            normalize_obs = normalize_obs
        )
    )

    actor_critic = ActorCritic(
        input_dim = env.obs_dim,
        actor_hidden_dim = [128, 128, 128],
        critic_hidden_dim = [128, 128, 128],
        action_dim = env.action_dim,
        activation = "elu",
        dropout = 0.0,
        norm = None,
        residual = False,
        init = "orthogonal",
        actor_gain = 2 ** 0.5,
        critic_gain = 2 ** 0.5,
        actor_output_gain = 0.01,
        critic_output_gain = 1.0,
        state_dependent = False # learned log_std starts at 0 -> std 1.0, rsl_rl's init_noise_std
    ).to(device)

    buffer = Buffer(
        num_steps = steps,
        num_envs = env.num_envs,
        obs_dim = env.obs_dim,
        action_dim = env.action_dim,
        device = device,
        gamma = gamma,
        lam = lam
    )

    ppo = PPO(
        env = env,
        actor_critic = actor_critic,
        buffer = buffer,
        steps = steps,
        epochs = epochs,
        num_minibatches = num_minibatches,
        lr = lr,
        clip_eps = clip_eps,
        c1_value = c1_value,
        c2_entropy = c2_entropy,
        max_grad_norm = max_grad_norm,
        desired_kl = desired_kl,
        clip_value = clip_value,
        normalize_value = normalize_value,
        normalize_obs = normalize_obs
    )

    pbar = tqdm(range(1, iters + 1), desc = "Training PPO")
    for it in pbar:
        ppo.rollout()
        ppo_stats = ppo.update()

        metrics = logger.log_iteration(
            step = it * steps * env.num_envs,
            ppo_stats = ppo_stats,
            buffer = buffer,
            extra = env.pop_episode_logs()
        )
        pbar.set_postfix({"avg_reward": f"{metrics['rollout/reward_mean']:.3f}", "kl": f"{ppo_stats['kl']:.4f}"})

    logger.close()

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model": actor_critic.state_dict(),
        "obs_mean": ppo.obs_rms.mean.cpu().numpy(),
        "obs_var": ppo.obs_rms.var.cpu().numpy(),
    }, f"checkpoints/{args.task}.pt")

    env.close()


if __name__ == "__main__":
    run_ppo()
    simulation_app.close()
