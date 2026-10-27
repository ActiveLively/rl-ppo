# built by Claude
import torch


class IsaacLabEnv:
    '''
    Isaac Lab env -> the get_obs()/step() interface PPO expects.
    Isaac Lab is already batched on the GPU and auto-resets the step an episode ends (like gym's SAME_STEP),
    so this only unwraps the "policy" obs group, clips actions and collects episode logs
    '''
    def __init__(self, env, clip_actions: float | None = None):
        self.env = env
        unwrapped = env.unwrapped
        self.device = unwrapped.device
        self.num_envs = unwrapped.num_envs
        self.obs_dim = unwrapped.single_observation_space["policy"].shape[0]
        self.action_dim = unwrapped.single_action_space.shape[0]
        self.clip_actions = clip_actions # action space is Box(-inf, inf), so bounds have to come from here

        obs, _ = self.env.reset()
        self.obs = obs["policy"]
        self.episode_logs = {}

    def get_obs(self):
        return self.obs

    def step(self, action):
        if self.clip_actions is not None:
            action = action.clamp(-self.clip_actions, self.clip_actions)
        obs, reward, terminated, truncated, extras = self.env.step(action)
        self.obs = obs["policy"]

        # extras["log"]: episode reward terms / termination counts, collected every step (same as rsl_rl's runner)
        for k, v in extras.get("log", {}).items():
            self.episode_logs.setdefault(k, []).append(v.item() if torch.is_tensor(v) else float(v))
        return self.obs, reward, terminated, truncated

    def pop_metrics(self):
        logs = {f"episode/{k}": sum(v) / len(v) for k, v in self.episode_logs.items()}
        self.episode_logs = {}
        return logs

    def close(self):
        self.env.close()
