import numpy as np
import torch
import os


import gymnasium as gym
import gymnasium_robotics
from tqdm import tqdm

from actor_critic import ActorCritic
from ppo import PPO
from buffer import Buffer

gym.register_envs(gymnasium_robotics)

class VecEnv:
    def __init__(self, env_id, num_envs, device, seed=0):
        def make_env():
            env = gym.make(env_id)

            if isinstance(env.observation_space, gym.spaces.Dict):
                env = gym.wrappers.FlattenObservation(env)
            env = gym.wrappers.NormalizeObservation(env)
            return env

        self.envs = gym.vector.SyncVectorEnv(
            [make_env for _ in range(num_envs)],
            autoreset_mode = gym.vector.AutoresetMode.SAME_STEP,
        )
        self.device = device
        self.obs_dim = self.envs.single_observation_space.shape[0]
        self.action_dim = self.envs.single_action_space.shape[0]
        self.action_low = torch.as_tensor(self.envs.single_action_space.low, device=device)
        self.action_high = torch.as_tensor(self.envs.single_action_space.high, device=device)

        obs, _ = self.envs.reset(seed=seed)
        self.obs = self._to_tensor(obs)
        self.episode_successes = []

    def _to_tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device = self.device)
    def get_obs(self):
        return self.obs
    def step(self, action):
        env_action = action.clamp(self.action_low, self.action_high).cpu().numpy()
        obs, reward, terminated, truncated, info = self.envs.step(env_action)
        self.obs = self._to_tensor(obs)
        if "final_info" in info and "is_success" in info["final_info"]:
            ended = info["_final_info"]
            self.episode_successes.extend(info["final_info"]["is_success"][ended].tolist())
        return self.obs, reward, terminated, truncated
    def pop_success_rate(self):
        if not self.episode_successes:
            return None
        rate = sum(self.episode_successes) / len(self.episode_successes)
        self.episode_successes = []
        return rate
    

    

def run_ppo():
    # PPO parameters
    num_envs = 8
    timesteps = 2000000
    steps = 256
    epochs = 12
    minibatch_size = 256
    lr = 3e-4
    clip_eps = 0.2
    c1_value = 0.5
    c2_entropy = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = VecEnv("FetchReachDense-v4", num_envs, device)

    # Buffer parameters
    # GAE gamma and lambda
    gamma = 0.98
    lam = 0.95

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

        
        postfix = {"avg_reward": f"{buffer.rewards.mean().item():.3f}"}
        success_rate = env.pop_success_rate()
        if success_rate is not None:
            postfix["success"] = f"{success_rate:.2f}"
        pbar.set_postfix(postfix)

    # save weights + obs normalization stats (the policy was trained on normalized obs)
    os.makedirs("checkpoints", exist_ok=True)
    obs_rms = env.envs.envs[0].obs_rms # NormalizeObservation is the outermost wrapper
    torch.save({
        "model": actor_critic.state_dict(),
        "obs_mean": obs_rms.mean,
        "obs_var": obs_rms.var,
    }, "checkpoints/fetch_reach.pt")


if __name__ == "__main__":
    run_ppo()