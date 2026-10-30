import torch 

from .mjx_env import _to_torch

class ConditionWrapper:
    '''
    Condition wrapper that modifies the obs data that is fed to the actor and critic (or just critic) without touching PPO:
    
        _added_data() -> returns the data we want to add to obs
        _augment(self, obs) -> if condition has a string item to condition on, concatenate 

        __getattr__(self, name) -> passes the functions that come through 
    '''
    def __init__(self, env, condition: str | None = None, sensors: list[str] | None = None,
                 sensor_clip: dict[str, float] | None = None):
        self.env = env
        self.condition = condition
        self.num_envs = env.num_envs
        self.obs_dim = env.obs_dim
        self.action_dim = env.action_dim
        self.device = env.device
        self.sensor_idx, self.sensor_limit, self.added_obs_names = None, None, []
        if condition == "force":
            self.sensor_idx, self.sensor_limit, self.added_obs_names = self._sensor_columns(sensors, sensor_clip or {})
        self.raw_sensors = [] # readings before sensor_clip since the last pop_metrics()
        self.obs = self._augment(env.get_obs())
        self.obs_dim = self.obs.shape[-1] # update obs dim to after a possible _added_data()

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
        if self.condition == "force":
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
            return obs  
        return torch.cat([obs, self._added_data()], dim = -1)

    def get_obs(self):
        return self.obs

    def step(self, action):
        obs, reward, terminated, truncated = self.env.step(action)
        self.obs = self._augment(obs)
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
            metrics.update({f"obs/{name}_{stat}": v for name, v in zip(self.added_obs_names, values.tolist())})
        return metrics

    def __getattr__(self, name):
        return getattr(self.env, name)

    
