import torch 

from .mjx_env import _to_torch

class ConditionWrapper:
    '''
    Condition wrapper that hands PPO conditioning data (cond) next to the env's obs, for the actor and critic (or just critic):

        _added_data() -> returns the conditioning data
        _augment(self, obs) -> (obs, cond). obs passes through unchanged, ActorCritic decides how to use cond
        get_cond() -> cond for the current obs, [num_envs, cond_dim]. cond_dim = 0 without a condition

        __getattr__(self, name) -> passes the functions that come through 
    '''
    def __init__(self, env, condition: str | None = None, sensors: list[str] | None = None,
                 sensor_clip: dict[str, float] | None = None, monitor: bool = False, critic_only: bool = True):
        self.env = env
        self.condition = condition
        self.num_envs = env.num_envs
        self.obs_dim = env.obs_dim
        self.action_dim = env.action_dim
        self.device = env.device
        if monitor and condition is not None:
            raise ValueError(f"monitor is for runs with no condition, got: {condition}")
        self.critic_only = critic_only # only the critic gets cond, see make_actor_critic in the training script
        self.sensor_idx, self.sensor_limit, self.cond_names, self.sensor_names = None, None, [], []
        if condition == "force" or monitor: # monitor: log the sensors, don't hand them to PPO
            self.sensor_idx, self.sensor_limit, self.sensor_names = self._sensor_columns(sensors, sensor_clip or {})
        if condition == "force":
            self.cond_names = self.sensor_names
        self.raw_sensors = [] # readings before sensor_clip since the last pop_metrics()
        self.obs, self.cond = self._augment(env.get_obs())
        self.cond_dim = self.cond.shape[-1]

    def _sensor_columns(self, names, sensor_clip):
        '''
        sensor names -> their columns in data.sensordata, each column's limit and its name. None = every sensor
        sensor_clip: {sensor name: limit}, in the sensor's units, like a real sensor saturating. missing = no limit
        '''
        m = self.env.mj_model
        for name in sensor_clip:
            m.sensor(name) # KeyError on a typo, which would otherwise leave that sensor unclipped
        idx, limits, channels = [], [], []
        for key in (range(m.nsensor) if names is None else names):
            s = m.sensor(key)
            adr, dim = int(s.adr[0]), int(s.dim[0])
            idx.extend(range(adr, adr + dim))
            limits.extend([sensor_clip.get(s.name, float("inf"))] * dim)
            name = s.name or f"sensor{s.id}"
            channels.extend([name] if dim == 1 else [f"{name}_{'xyz'[i] if dim == 3 else i}" for i in range(dim)])
        return torch.tensor(idx, device = self.device), torch.tensor(limits, device = self.device), channels
    
    def _raw_data(self, data = None):
        if data is None:
            data = self.env.data
        if self.sensor_idx is not None:
            extra = _to_torch(data.sensordata)[..., self.sensor_idx]
        elif self.condition == "torque":
            extra = _to_torch(data.actuator_force)
        else:
            raise ValueError(f"adding unknown condition: {self.condition}")
        return torch.nan_to_num(extra, nan = 0.0, posinf = 0.0, neginf = 0.0)

    def _added_data(self, data = None):
        extra = self._raw_data(data)
        if self.sensor_limit is not None:
            extra = torch.clamp(extra, -self.sensor_limit, self.sensor_limit)
        return extra

    def _augment(self, obs):
        if self.condition is None:
            return obs, obs[..., :0] # empty cond, [num_envs, 0]
        return obs, self._added_data()

    def get_obs(self):
        return self.obs

    def get_cond(self):
        return self.cond

    def step(self, action):
        obs, reward, terminated, truncated = self.env.step(action)
        self.obs, self.cond = self._augment(obs)
        if self.sensor_idx is not None:
            self.raw_sensors.append(self._raw_data())
        return self.obs, reward, terminated, truncated

    def pop_metrics(self):
        '''the wrapped env's metrics + per channel stats of the readings before sensor_clip'''
        metrics = self.env.pop_metrics()
        if not self.raw_sensors:
            return metrics
        raw = torch.cat(self.raw_sensors) # [steps * num_envs, channels]
        self.raw_sensors = []
        size = raw.abs() # spikes go both ways, and the limit is symmetric
        stats = {
            "mean": raw.mean(dim = 0),
            "p99": size.quantile(0.99, dim = 0),
            "max": size.amax(dim = 0),
            "clip_frac": (size > self.sensor_limit).float().mean(dim = 0)
        }
        for stat, values in stats.items():
            metrics.update({f"obs/{name}_{stat}": v for name, v in zip(self.sensor_names, values.tolist())})
        return metrics

    def __getattr__(self, name):
        return getattr(self.env, name)

    
