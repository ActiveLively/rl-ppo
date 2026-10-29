import torch 

from .mjx_env import _to_torch

class ConditionWrapper:
    '''
    Condition wrapper that modifies the obs data that is fed to the actor and critic (or just critic) without touching PPO:
    
        _added_data() -> returns the data we want to add to obs
        _augment(self, obs) -> if condition has a string item to condition on, concatenate 

        __getattr__(self, name) -> passes the functions that come through 
    '''
    def __init__(self, env, condition: str | None = None, sensors: list[str] | None = None):
        self.env = env
        self.condition = condition
        self.num_envs = env.num_envs
        self.obs_dim = env.obs_dim
        self.action_dim = env.action_dim
        self.device = env.device
        self.sensor_idx = self._sensor_idx(sensors) if condition == "force" else None
        self.obs = self._augment(env.get_obs())
        self.obs_dim = self.obs.shape[-1] # update obs dim to after a possible _added_data()

    def _sensor_idx(self, names):
        '''
        sensor names -> their columns in data.sensordata. None = every sensor
        Link the names with the sensor idxs
        '''
        m = self.env.mj_model
        if names is None:
            return torch.arange(m.nsensordata, device = self.device)
        idx = []
        for name in names:
            s = m.sensor(name) 
            idx.extend(range(int(s.adr[0]), int(s.adr[0]) + int(s.dim[0])))
        return torch.tensor(idx, device = self.device)
    
    def _added_data(self, data = None):
        if data is None:
            data = self.env.data
        if self.condition == "force":
            extra = _to_torch(data.sensordata)[..., self.sensor_idx]
        elif self.condition == "torque":
            extra = _to_torch(data.actuator_force)
        else:
            raise ValueError(f"adding unknown condition: {self.condition}")
        return torch.nan_to_num(extra, nan = 0.0, posinf = 0.0, neginf = 0.0)

    def _augment(self, obs):
        if self.condition is None:
            return obs  
        return torch.cat([obs, self._added_data()], dim = -1)

    def get_obs(self):
        return self.obs

    def step(self, action):
        obs, reward, terminated, truncated = self.env.step(action)
        self.obs = self._augment(obs)
        return self.obs, reward, terminated, truncated

    def __getattr__(self, name):
        return getattr(self.env, name)

   
