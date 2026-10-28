import argparse
# from omegaconf import OmegaConf

import torch

from rl_ppo import ActorCritic, Buffer, PPO, Runner, TrainingLogger
from rl_ppo.envs.mjx_env import MJXEnv
from rl_ppo.logger import WandbWriter


parser = argparse.ArgumentParser()

parser.add_argument("--env", default="CartpoleBalance") # any name in mujoco_playground.registry.ALL_ENVS
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--iters", type=int, default=300)
parser.add_argument("--episode_length", type=int, default=None) # None = the env's default
parser.add_argument("--impl", default=None) # Playground >= 0.1: "jax" on a Mac / non-NVIDIA, "warp" needs an NVIDIA GPU
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--wandb_project", default="rl-ppo")
parser.add_argument("--wandb_group", default=None) # e.g. one group per experiment, one run per seed

# cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
# cfg = OmegaConf.to_container(cfg, resolve=True)

args = parser.parse_args()



def run_ppo():
    # PPO parameters (rsl_rl-style, gamma from Playground's dm_control configs)
    steps = 32
    epochs = 5
    num_minibatches = 4
    lr = 1e-3
    clip_eps = 0.2
    c1_value = 1.0
    c2_entropy = 0.01
    max_grad_norm = 1.0
    desired_kl = 0.01
    clip_value = True
    normalize_value = True # 1000-step episodes with gamma 0.995 -> returns in the 100s
    normalize_obs = True
    gamma = 0.995
    lam = 0.95

    env = MJXEnv(
        args.env, args.num_envs, episode_length=args.episode_length,
        config_overrides={"impl": args.impl} if args.impl else None
    )
    device = env.device

    run_name = f"mjx_{args.env}"
    config = dict(
        env_id = args.env, num_envs = env.num_envs, iters = args.iters, steps = steps, epochs = epochs,
        num_minibatches = num_minibatches, lr = lr, clip_eps = clip_eps, c1_value = c1_value,
        c2_entropy = c2_entropy, gamma = gamma, lam = lam, max_grad_norm = max_grad_norm,
        desired_kl = desired_kl, clip_value = clip_value, normalize_value = normalize_value,
        normalize_obs = normalize_obs, impl = args.impl
    )

    writers = ["jsonl"]
    if args.wandb:
        writers.append(WandbWriter(project=args.wandb_project, name=run_name, group=args.wandb_group, config=config))

    logger = TrainingLogger(
        log_dir = "runs",
        run_name = run_name,
        writers = writers,
        config = config
    )

    actor_critic = ActorCritic(
        input_dim = env.obs_dim,
        actor_hidden_dim = [256, 256],
        critic_hidden_dim = [256, 256],
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
        state_dependent = False
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

    runner = Runner(env, ppo, logger)
    runner.train(iters = args.iters)
    logger.close()

    runner.save(f"checkpoints/{run_name}.pt")


if __name__ == "__main__":
    run_ppo()
