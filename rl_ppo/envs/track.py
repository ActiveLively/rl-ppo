class TrackWrapper:
    '''
    VecEnv wrapper that feeds a sim-side collector (e.g. sim_logging.SimDataCollector) without touching PPO:
        step()        -> collector.on_env_step(terminated, truncated), right after the env step
        pop_metrics() -> the env's episode stats + collector.end_iteration(global_step)
    Runner calls pop_metrics() once per iteration, so the force / effort scalars land in the same
    metrics.jsonl line as the PPO diagnostics
    '''
    def __init__(self, env, collector):
        self.env = env
        self.collector = collector
        self.num_envs = env.num_envs
        self.obs_dim = env.obs_dim
        self.action_dim = env.action_dim
        self.device = env.device
        self.global_step = 0

    def get_obs(self):
        return self.env.get_obs()

    def step(self, action):
        obs, reward, terminated, truncated = self.env.step(action)
        # functional sims (MJXEnv) hand over their state, the others are read in place and have no .data
        self.collector.on_env_step(terminated, truncated, data = getattr(self.env, "data", None))
        self.global_step += self.num_envs
        return obs, reward, terminated, truncated

    def pop_metrics(self):
        return {**self.env.pop_metrics(), **self.collector.end_iteration(self.global_step)}

    def __getattr__(self, name):
        return getattr(self.env, name) # mj_model, data, ... of the wrapped env
