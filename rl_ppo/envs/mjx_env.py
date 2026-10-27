import jax
import torch
from torch.utils import dlpack as torch_dlpack
from mujoco_playground import registry, wrapper


def _to_torch(x):
    '''jax array -> torch tensor on the same device, zero-copy'''
    return torch_dlpack.from_dlpack(x)


def _to_jax(x):
    '''torch tensor -> jax array on the same device, zero-copy'''
    return jax.dlpack.from_dlpack(x.contiguous())


class MJXEnv:
    '''
    MuJoCo Playground env (MJX, physics in JAX) -> VecEnv. all envs step in one jit-compiled, vmapped call
    on the GPU (CPU on a Mac), and tensors cross between JAX and torch through dlpack without copies.
    wrap_for_brax_training adds the time limit + same-step auto-reset:
      state.done               = terminated or timed out
      state.info["truncation"] = timed out and not terminated
    '''
    def __init__(
            self,
            env_name: str,
            num_envs: int,
            seed: int = 0,
            episode_length: int | None = None,
            action_repeat: int = 1,
            impl: str | None = None,
            obs_key: str = "state",
            clip_actions: float | None = 1.0,
            full_reset: bool = False
        ):
        '''
        episode_length: time limit in env steps, None = the env's default config
        impl:           MJX backend, "jax" (any device) or "warp" (NVIDIA only). None = the env's default.
                        Playground >= 0.1 only, older versions are always "jax"
        obs_key:        which obs group the policy sees when obs is a dict (locomotion: "state" / "privileged_state")
        clip_actions:   Playground actions are in [-1, 1], the env scales them to actuator ranges
        full_reset:     False = done envs restore their cached first state (Playground default, fast but every env
                        always restarts from the same pose). True = sample a fresh random reset each episode.
                        Playground >= 0.1 only
        '''
        env_cfg = registry.get_default_config(env_name)
        env = registry.load(env_name, config = env_cfg, config_overrides = {"impl": impl} if impl else None)
        wrap_kwargs = {"full_reset": True} if full_reset else {} # only passed when set, so Playground 0.0.x still works
        self.env = wrapper.wrap_for_brax_training(
            env,
            episode_length = episode_length or env_cfg.episode_length,
            action_repeat = action_repeat,
            **wrap_kwargs
        )
        self._reset = jax.jit(self.env.reset)
        self._step = jax.jit(self.env.step)

        self.num_envs = num_envs
        self.obs_key = obs_key
        self.clip_actions = clip_actions
        self.action_dim = env.action_size

        self.state = self._reset(jax.random.split(jax.random.PRNGKey(seed), num_envs)) # one key per env
        self.obs = self._get_obs()
        self.obs_dim = self.obs.shape[-1]
        self.device = self.obs.device

        # brax's EpisodeWrapper keeps running episode totals in info["episode_metrics"] (sum_reward, length, env metrics).
        # summed on device for the episodes that end, so step() never syncs with the CPU
        self.metric_sums = {}
        self.episodes_done = torch.zeros((), device = self.device)

    def _get_obs(self):
        obs = self.state.obs
        if isinstance(obs, dict):
            obs = obs[self.obs_key]
        return _to_torch(obs)

    def get_obs(self):
        return self.obs

    def step(self, action):
        if self.clip_actions is not None:
            action = action.clamp(-self.clip_actions, self.clip_actions)
        self.state = self._step(self.state, _to_jax(action))
        self.obs = self._get_obs()

        reward = _to_torch(self.state.reward)
        done = _to_torch(self.state.done)
        truncated = _to_torch(self.state.info["truncation"])
        terminated = done * (1 - truncated)

        # on the step an episode ends, episode_metrics still holds its totals (they are zeroed on the next step)
        for k, v in self.state.info["episode_metrics"].items():
            self.metric_sums[k] = self.metric_sums.get(k, 0.0) + (_to_torch(v) * done).sum()
        self.episodes_done += done.sum()
        return self.obs, reward, terminated, truncated

    def pop_metrics(self):
        n = self.episodes_done.item()
        if n == 0:
            return {}
        logs = {f"episode/{k}": (v / n).item() for k, v in self.metric_sums.items()}
        self.metric_sums = {}
        self.episodes_done.zero_()
        return logs
