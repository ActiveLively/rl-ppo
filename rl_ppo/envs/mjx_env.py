import jax
import torch
from brax.envs.wrappers import training as brax_training
from torch.utils import dlpack as torch_dlpack
from mujoco_playground import MjxEnv, registry, wrapper


def _to_torch(x):
    '''jax array -> torch tensor on the same device, zero-copy'''
    return torch_dlpack.from_dlpack(x)


def _to_jax(x):
    '''torch tensor -> jax array on the same device, zero-copy'''
    return jax.dlpack.from_dlpack(x.contiguous())


_PRESERVE = "AutoResetWrapper_preserve_info" # the one info key Playground's auto-reset carries across a full reset


class _KeepEpisodeInfo(wrapper.Wrapper):
    '''
    sits between EpisodeWrapper and the auto-reset. with full_reset, the auto-reset swaps a finished env's info for
    the new episode's info on the same step it ends, wiping truncation and episode_metrics before step() reads them.
    copies both under _PRESERVE, which survives the swap. set in reset() too: the auto-reset tree-maps the reset info
    against the step info, so both need the key
    '''
    def _keep(self, state):
        preserved = dict(state.info.get(_PRESERVE, {})) # merge, in case the env keeps its own curriculum state here
        preserved["rl_ppo"] = {
            "truncation": state.info["truncation"],
            "episode_metrics": dict(state.info["episode_metrics"]) # copy: EpisodeWrapper updates this dict in place
        }
        state.info[_PRESERVE] = preserved
        return state

    def reset(self, rng):
        return self._keep(self.env.reset(rng))

    def step(self, state, action):
        return self._keep(self.env.step(state, action))


class MJXEnv:
    '''
    MuJoCo Playground env (MJX, physics in JAX) -> VecEnv. all envs step in one jit-compiled, vmapped call
    on the GPU (CPU on a Mac), and tensors cross between JAX and torch through dlpack without copies.
    same wrappers as Playground's wrap_for_brax_training (vmap -> time limit -> same-step auto-reset), plus
    _KeepEpisodeInfo so the episode-end values survive a full reset:
      state.done               = terminated or timed out
      truncation               = timed out and not terminated
    '''
    def __init__(
            self,
            env: str | MjxEnv,
            num_envs: int,
            seed: int = 0,
            episode_length: int | None = None,
            action_repeat: int = 1,
            config_overrides: dict | None = None,
            obs_key: str = "state",
            clip_actions: float | None = 1.0,
            full_reset: bool = False,
            randomize_start: bool = False
        ):
        '''
        env:              a Playground registry name, or an already built Playground env (e.g. your own task)
        episode_length:   time limit in env steps, None = the env's config
        config_overrides: registry names only, e.g. {"action_scale": 0.01}. {"impl": "warp"} picks the MJX backend
                          ("jax" on any device, "warp" NVIDIA only, Playground >= 0.1)
        obs_key:          which obs group the policy sees when obs is a dict (locomotion: "state" / "privileged_state")
        clip_actions:     Playground actions are in [-1, 1], the env scales them to actuator ranges
        full_reset:       False = done envs restore their cached first state (Playground default, fast but every env
                          always restarts from the same pose). True = sample a fresh random reset each episode.
                          Playground >= 0.1 only
        randomize_start:  True = begin each env partway through its first episode's time limit, so resets are spread
                          across the episode instead of landing on the same step. only the first episode is cut short
        '''
        if isinstance(env, str):
            env = registry.load(env, config_overrides = config_overrides)
        elif config_overrides:
            raise ValueError("config_overrides only applies to registry names, configure a built env yourself")
        if episode_length is None:
            episode_length = env.unwrapped._config.episode_length # every Playground env keeps its config here

        wrapped = brax_training.VmapWrapper(env)
        wrapped = brax_training.EpisodeWrapper(wrapped, episode_length, action_repeat)
        wrapped = _KeepEpisodeInfo(wrapped)
        reset_kwargs = {"full_reset": True} if full_reset else {} # only passed when set, so Playground 0.0.x still works
        self.env = wrapper.BraxAutoResetWrapper(wrapped, **reset_kwargs)
        self._reset = jax.jit(self.env.reset)
        self._step = jax.jit(self.env.step)

        self.num_envs = num_envs
        self.obs_key = obs_key
        self.clip_actions = clip_actions
        self.action_dim = env.action_size

        key = jax.random.PRNGKey(seed)
        self.state = self._reset(jax.random.split(key, num_envs)) # one key per env
        if randomize_start:
            start = jax.random.randint(jax.random.fold_in(key, 1), (num_envs,), 0, episode_length)
            self.state.info["steps"] = self.state.info["steps"] + start
        self.obs = self._get_obs()
        self.obs_dim = self.obs.shape[-1]
        self.device = self.obs.device

        # brax's EpisodeWrapper keeps running episode totals in info["episode_metrics"] (sum_reward, length, env metrics).
        # summed on device for the episodes that end, so step() never syncs with the CPU
        self.metric_sums = {}
        self.episodes_done = torch.zeros((), device = self.device)

    @property
    def mj_model(self):
        '''mujoco.MjModel (CPU), e.g. for body / geom ids'''
        return self.env.mj_model

    @property
    def mjx_model(self):
        '''mjx.Model, the model the jitted step runs'''
        return self.env.mjx_model

    @property
    def data(self):
        '''current mjx.Data, batched: every array has num_envs as its first dim. JAX arrays on the sim device'''
        return self.state.data

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
        episode = self.state.info[_PRESERVE]["rl_ppo"] # this step's values, even for envs that were just reset
        truncated = _to_torch(episode["truncation"])
        terminated = done * (1 - truncated)

        # on the step an episode ends, episode_metrics still holds its totals
        for k, v in episode["episode_metrics"].items():
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
