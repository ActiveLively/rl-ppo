import torch
import gymnasium as gym
import gymnasium_robotics

from rl_ppo import ActorCritic, Buffer, PPO, Runner, TrainingLogger
from rl_ppo.envs.gym_env import GymVecEnv

gym.register_envs(gymnasium_robotics)


def run_ppo():
    # PPO parameters
    num_envs = 8
    timesteps = 2000000
    steps = 256
    epochs = 12
    num_minibatches = 8 # 256 steps * 8 envs / 8 = 256 samples per minibatch
    lr = 3e-4
    clip_eps = 0.2
    c1_value = 0.5
    c2_entropy = 0.001
    max_grad_norm = 0.5
    desired_kl = None # e.g. 0.01 for adaptive lr, None = fixed lr
    clip_value = False
    normalize_value = False
    normalize_obs = True
    device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = GymVecEnv("FetchReachDense-v4", num_envs, device)

    # Buffer parameters
    # GAE gamma and lambda
    gamma = 0.98
    lam = 0.95

    # logging: PPO diagnostics + rollout stats -> runs/<run_name>/metrics.jsonl
    logger = TrainingLogger(
        log_dir = "runs",
        run_name = "fetch_reach",
        writers = ("jsonl",), # add "tensorboard" after `pip install tensorboard`
        config = dict(
            env_id = "FetchReachDense-v4", num_envs = num_envs, timesteps = timesteps, steps = steps,
            epochs = epochs, num_minibatches = num_minibatches, lr = lr, clip_eps = clip_eps,
            c1_value = c1_value, c2_entropy = c2_entropy, gamma = gamma, lam = lam,
            max_grad_norm = max_grad_norm, desired_kl = desired_kl, clip_value = clip_value,
            normalize_value = normalize_value, normalize_obs = normalize_obs
        )
    )

    actor_critic = ActorCritic(
        input_dim = env.obs_dim,
        actor_hidden_dim = [256,256],
        critic_hidden_dim = [256, 256],
        action_dim = env.action_dim,
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

    # iterations loop, rollout (collect data + compute GAE) -> update (evaluate + loss calculation + optimizer)
    runner = Runner(env, ppo, logger)
    runner.train(iters = timesteps // (steps * num_envs))
    logger.close()

    # save weights + obs normalization stats (the policy was trained on normalized obs)
    runner.save("checkpoints/fetch_reach.pt")


if __name__ == "__main__":
    run_ppo()