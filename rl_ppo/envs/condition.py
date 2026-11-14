from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .mjx_env import _to_torch


@dataclass
class CondSignal:
    '''
    one named piece of conditioning data, e.g. a wrist F/T sensor or a value the actor can't observe

    read:  fn(state) -> array (jax or torch) of the env's batched state, [num_envs, ...]. given one unbatched
           state (a video env) it returns [...]. any trailing shape, it is flattened to columns
    names: one per column, for the logs. None = <signal name>_<i>
    limit: readings are clamped to +-limit (one value, or one per column), like a real sensor saturating. None = no limit
    '''
    read: Callable
    names: list[str] | None = None
    limit: float | list[float] | None = None


def sensor_signal(mj_model, sensors, limits = None):
    '''
    MuJoCo sensors by name -> their columns of data.sensordata, named <sensor>_x / _y / _z
    limits: {sensor name: limit}, in the sensor's units. missing = no limit
    '''
    limits = limits or {}
    for name in limits:
        mj_model.sensor(name) # KeyError on a typo, which would otherwise leave that sensor unclipped
    idx, limit, channels = [], [], []
    for key in sensors:
        s = mj_model.sensor(key)
        adr, dim = int(s.adr[0]), int(s.dim[0])
        idx.extend(range(adr, adr + dim))
        limit.extend([limits.get(s.name, float("inf"))] * dim)
        name = s.name or f"sensor{s.id}"
        channels.extend([name] if dim == 1 else [f"{name}_{'xyz'[i] if dim == 3 else i}" for i in range(dim)])
    idx = np.array(idx)
    return CondSignal(read = lambda state: state.data.sensordata[..., idx], names = channels, limit = limit)


def data_signal(field, names = None, limit = None):
    '''an mjx.Data field, e.g. "actuator_force" (the joint torques)'''
    return CondSignal(read = lambda state: getattr(state.data, field), names = names, limit = limit)


def info_signal(key, names = None, limit = None):
    '''an env state.info entry, e.g. a randomized value the actor can't observe'''
    return CondSignal(read = lambda state: state.info[key], names = names, limit = limit)


class ConditionWrapper:
    '''
    Condition wrapper that hands PPO conditioning data (cond) next to the env's obs, picked by name from `signals`:

        cond_all:    signals for the actor and the critic
        cond_critic: signals for the critic alone
        monitor:     signals whose readings (before their limit) are logged per channel, handed to PPO or not

    cond's columns are [cond_all..., cond_critic...] in the order given. the critic gets all of them, the actor the
    first actor_cond_dim (see make_actor_critic in the training script), so one run can condition both networks on
    some signals and only the critic on others.

        cond_of(state) -> cond of the current state, or of a state from another env of the same task
        get_cond() -> cond for the current obs, [num_envs, cond_dim]. cond_dim = 0 without any signal
        obs passes through unchanged, ActorCritic decides how to use cond

        __getattr__(self, name) -> passes the functions that come through
    '''
    def __init__(self, env, signals: dict[str, CondSignal], cond_all: list[str] | None = None,
                 cond_critic: list[str] | None = None, monitor: list[str] | None = None):
        self.env = env
        self.signals = signals
        self.num_envs = env.num_envs
        self.obs_dim = env.obs_dim
        self.action_dim = env.action_dim
        self.device = env.device
        self.cond_all = list(cond_all or [])
        self.cond_critic = list(cond_critic or [])
        self.monitor = list(monitor or [])
        both = set(self.cond_all) & set(self.cond_critic)
        if both:
            raise ValueError(f"in cond_all and cond_critic, pick one: {sorted(both)}")
        self.used = list(dict.fromkeys(self.cond_all + self.cond_critic)) # cond's signals, in column order
        unknown = [n for n in self.used + self.monitor if n not in signals]
        if unknown:
            raise ValueError(f"unknown condition signals {unknown}, choose from {sorted(signals)}")

        # per signal: its limit as a tensor and its column names. read once here for the widths
        self.limits, names = {}, {}
        for name in dict.fromkeys(self.used + self.monitor):
            sig, width = signals[name], self._raw(name).shape[-1]
            limit = float("inf") if sig.limit is None else sig.limit
            self.limits[name] = torch.as_tensor(limit, dtype = torch.float32, device = self.device).expand(width)
            names[name] = list(sig.names) if sig.names else ([name] if width == 1 else [f"{name}_{i}" for i in range(width)])
            if len(names[name]) != width:
                raise ValueError(f"signal '{name}' has {width} columns but {len(names[name])} names")
        self.cond_names = [c for n in self.used for c in names[n]]
        self.monitor_names = [c for n in self.monitor for c in names[n]]
        self.monitor_limit = torch.cat([self.limits[n] for n in self.monitor]) if self.monitor else None
        # how many of cond's leading columns the actor gets
        self.actor_cond_dim = sum(len(names[n]) for n in self.cond_all)

        self.raw_monitor = [] # monitored readings before their limit since the last pop_metrics()
        self.obs = env.get_obs()
        self.cond = self._cond(self.obs)
        self.cond_dim = self.cond.shape[-1]

    def _raw(self, name, state = None):
        '''a signal's reading before its limit, [num_envs, width] ([1, width] for one unbatched state)'''
        single = state is not None and state.done.ndim == 0
        x = self.signals[name].read(self.env.state if state is None else state)
        if not isinstance(x, torch.Tensor):
            x = _to_torch(x)
        x = x.float().reshape(1 if single else x.shape[0], -1)
        return torch.nan_to_num(x, nan = 0.0, posinf = 0.0, neginf = 0.0)

    def _clip(self, name, x):
        limit = self.limits[name].to(x.device)
        return torch.clamp(x, -limit, limit)

    def cond_of(self, state = None):
        '''
        cond of the current state, or of a state from another env of the same task (an eval / video env next to
        training): batched -> [num_envs, cond_dim], one unbatched state -> [1, cond_dim]. None without any signal
        '''
        if not self.used:
            return None
        return torch.cat([self._clip(n, self._raw(n, state)) for n in self.used], dim = -1)

    def _cond(self, obs, raw = None):
        '''cond for the current obs. raw: this step's readings, when they are already read. [num_envs, 0] without any'''
        if not self.used:
            return obs[..., :0]
        if raw is None:
            return self.cond_of()
        return torch.cat([self._clip(n, raw[n]) for n in self.used], dim = -1)

    def get_obs(self):
        return self.obs

    def get_cond(self):
        return self.cond

    def step(self, action):
        self.obs, reward, terminated, truncated = self.env.step(action)
        raw = {n: self._raw(n) for n in self.limits} # every signal in use, read once
        self.cond = self._cond(self.obs, raw)
        if self.monitor:
            self.raw_monitor.append(torch.cat([raw[n] for n in self.monitor], dim = -1))
        return self.obs, reward, terminated, truncated

    def pop_metrics(self):
        '''the wrapped env's metrics + per channel stats of the monitored readings before their limit'''
        metrics = self.env.pop_metrics()
        if not self.raw_monitor:
            return metrics
        raw = torch.cat(self.raw_monitor) # [steps * num_envs, channels]
        self.raw_monitor = []
        size = raw.abs() # spikes go both ways, and the limit is symmetric
        stats = {
            "mean": raw.mean(dim = 0),
            "p99": size.quantile(0.99, dim = 0),
            "max": size.amax(dim = 0),
            "clip_frac": (size > self.monitor_limit).float().mean(dim = 0)
        }
        for stat, values in stats.items():
            metrics.update({f"obs/{name}_{stat}": v for name, v in zip(self.monitor_names, values.tolist())})
        return metrics

    def __getattr__(self, name):
        return getattr(self.env, name)
