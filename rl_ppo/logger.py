import json
import os
import time

import torch


# ---------------------------------------------------------------------------------------------
# PPO training logger: algorithm diagnostics + rollout stats only. sim data (forces, torques,
# raw per-step recording) comes from a separate collector and is passed in through `extra`
# ---------------------------------------------------------------------------------------------
class MeanTracker:
    '''averages scalar diagnostics over minibatches, e.g. inside PPO.update()'''
    def __init__(self):
        self.sums, self.counts = {}, {}

    def add(self, **metrics):
        for k, v in metrics.items():
            v = v.item() if torch.is_tensor(v) else float(v)
            self.sums[k] = self.sums.get(k, 0.0) + v
            self.counts[k] = self.counts.get(k, 0) + 1

    def means(self):
        return {k: self.sums[k] / self.counts[k] for k in self.sums}


@torch.no_grad()
def rollout_stats(buffer):
    '''stats from a filled Buffer (after compute_advantage_return). advantages are already normalized'''
    values, returns = buffer.values.flatten(), buffer.returns.flatten()
    var_returns = returns.var()
    explained_var = 1 - (returns - values).var() / var_returns if var_returns > 0 else torch.tensor(float("nan"))
    return {
        "rollout/reward_mean": buffer.rewards.mean().item(),
        "rollout/reward_std": buffer.rewards.std().item(),
        "rollout/value_mean": values.mean().item(),
        "rollout/return_mean": returns.mean().item(),
        "rollout/explained_variance": explained_var.item(),
        "rollout/terminated_frac": buffer.terminateds.mean().item(),
        "rollout/truncated_frac": buffer.truncateds.mean().item(),
    }


# ---------------------------------------------------------------------------------------------
# Writers: anything with write(step, metrics) and close()
# ---------------------------------------------------------------------------------------------
class JSONLWriter:
    '''one json object per line -> pandas.read_json(path, lines=True)'''
    def __init__(self, log_dir):
        self.f = open(os.path.join(log_dir, "metrics.jsonl"), "a")

    def write(self, step, metrics):
        self.f.write(json.dumps({"step": step, **metrics}) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


class TensorBoardWriter:
    def __init__(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter  # needs `pip install tensorboard`
        self.w = SummaryWriter(log_dir)

    def write(self, step, metrics):
        for k, v in metrics.items():
            self.w.add_scalar(k, v, step)

    def close(self):
        self.w.close()


class WandbWriter:
    '''Weights & Biases; joins the caller's run if one is already active, otherwise starts (and later finishes) its own.
    not in WRITERS since it needs a project/config instead of log_dir: pass an instance in TrainingLogger(writers=...)'''
    def __init__(self, project=None, name=None, config=None, group=None, tags=None, **init_kwargs):
        import wandb  # needs `pip install wandb`
        self.owns_run = wandb.run is None
        if self.owns_run:
            self.run = wandb.init(project=project, name=name, config=config, group=group, tags=tags, **init_kwargs)
        else:
            self.run = wandb.run
            if config:
                self.run.config.update({"ppo": config}, allow_val_change=True)  # namespaced, no clash with the caller's keys

        # x-axis = env steps, not wandb's internal log counter, so other code can log to the same run anytime
        self.run.define_metric("env_step")
        self.run.define_metric("*", step_metric="env_step")

    def write(self, step, metrics):
        self.run.log({"env_step": step, **metrics})

    def close(self):
        if self.owns_run:  # the caller finishes runs it started
            self.run.finish()


WRITERS = {"jsonl": JSONLWriter, "tensorboard": TensorBoardWriter}


# ---------------------------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------------------------
class TrainingLogger:
    '''metrics.jsonl: one line per iteration with PPO diagnostics, rollout stats and any `extra` scalars'''
    def __init__(self, log_dir="runs", run_name=None, writers=("jsonl",), config=None):
        run_name = run_name or time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(log_dir, run_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.writers = [WRITERS[w](self.log_dir) if isinstance(w, str) else w for w in writers]
        self.start_time = time.time()

        if config is not None:
            with open(os.path.join(self.log_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2, default=str)

    def log_iteration(self, step, ppo_stats=None, buffer=None, extra=None):
        metrics = {}
        if ppo_stats:
            metrics.update({f"ppo/{k}": v for k, v in ppo_stats.items()})
        if buffer is not None:
            metrics.update(rollout_stats(buffer))
        if extra:
            metrics.update({k: v for k, v in extra.items() if v is not None})
        metrics["time/elapsed_s"] = time.time() - self.start_time
        metrics["time/steps_per_s"] = step / max(metrics["time/elapsed_s"], 1e-8)

        for w in self.writers:
            w.write(step, metrics)
        return metrics

    def close(self):
        for w in self.writers:
            w.close()
