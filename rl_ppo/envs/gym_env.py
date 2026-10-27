import torch
import gymnasium as gym


class GymVecEnv:
    '''
    gymnasium env id -> VecEnv. SyncVectorEnv steps on the CPU, tensors are moved to `device`.
    obs are raw, PPO normalizes them (normalize_obs)
    '''
    def __init__(self, env_id, num_envs, device, seed=0):
        def make_env():
            env = gym.make(env_id)

            if isinstance(env.observation_space, gym.spaces.Dict):
                env = gym.wrappers.FlattenObservation(env)
            return env

        self.envs = gym.vector.SyncVectorEnv(
            [make_env for _ in range(num_envs)],
            autoreset_mode = gym.vector.AutoresetMode.SAME_STEP,
        )
        self.num_envs = num_envs
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
        return self.obs, self._to_tensor(reward), self._to_tensor(terminated), self._to_tensor(truncated)

    def pop_metrics(self):
        '''success rate of the episodes that ended since the last call (goal envs like Fetch only)'''
        if not self.episode_successes:
            return {}
        rate = sum(self.episode_successes) / len(self.episode_successes)
        self.episode_successes = []
        return {"rollout/success_rate": rate}
