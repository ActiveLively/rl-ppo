import torch 
from actor_critic import ActorCritic
from buffer import Buffer
from logger import MeanTracker
from normalization import RunningMeanStd

# bounds for the adaptive KL learning rate
MIN_LR = 1e-5
MAX_LR = 1e-2

class PPO():
    def __init__(
            self,
            env,
            actor_critic: ActorCritic, 
            buffer: Buffer,
            steps: int,
            epochs: int,
            minibatch_size: int,
            lr: float = 3e-4,
            clip_eps: float = 0.2,
            c1_value: float = 0.5,
            c2_entropy: float = 0.0,
            max_grad_norm: float = 0.5,
            desired_kl: float | None = None,
            clip_value: bool = False,
            normalize_value: bool = False
        ):
        '''
        max_grad_norm:   global grad-norm clip
        desired_kl:      adaptive lr (rsl_rl / skrl KLAdaptiveLR): lr /1.5 if kl > 2*desired_kl, lr *1.5 if kl < desired_kl/2.
                         kl is the exact Gaussian KL(old || new). None = fixed lr
        clip_value:      PPO-style value clipping, value can move at most clip_eps from its rollout prediction
        normalize_value: critic predicts returns normalized by running mean/std (skrl value_preprocessor).
                         buffer still holds values/returns in reward scale, so GAE and logging are unchanged
        '''
        self.env = env
        self.actor_critic = actor_critic
        self.buffer = buffer
        self.steps = steps
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.lr = lr
        self.clip_eps = clip_eps
        self.c1_value = c1_value
        self.c2_entropy = c2_entropy
        self.max_grad_norm = max_grad_norm
        self.desired_kl = desired_kl
        self.clip_value = clip_value

        device = next(self.actor_critic.parameters()).device
        self.value_rms = RunningMeanStd(shape = (), device = device) if normalize_value else None

        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr = self.lr)

    def _denormalize_value(self, value):
        '''critic output -> reward scale'''
        return self.value_rms.denormalize(value) if self.value_rms is not None else value

    @torch.no_grad()
    def rollout(self):
        self.buffer.reset()
        obs = self.env.get_obs()
        
        for i in range(self.steps):
            action, log_prob, value, mu, sigma = self.actor_critic.act(obs)
            next_obs, reward, terminated, truncated = self.env.step(action)
            self.buffer.add(obs, action, log_prob, reward, terminated, truncated, self._denormalize_value(value), mu, sigma)

            obs = next_obs

        # bootstrap value of the state after the last collected step -> [num_envs]
        next_values = self._denormalize_value(self.actor_critic.value(obs))

        self.buffer.compute_advantage_return(next_values)

        # update return stats once per rollout, they stay fixed for all epochs of update()
        if self.value_rms is not None:
            self.value_rms.update(self.buffer.returns)


    def update(self):
        stats = MeanTracker()
        for epoch in range(self.epochs):
            for mb in self.buffer.get_minibatches(minibatch_size = self.minibatch_size):
                b_obs = mb["obs"]
                b_actions = mb["actions"]
                b_log_probs = mb["log_probs"]
                b_advantages = mb["advantages"]
                b_returns = mb["returns"]
                b_values = mb["values"]
                b_mu = mb["mu"]
                b_sigma = mb["sigma"]

                # value loss is computed in the critic's output space
                if self.value_rms is not None:
                    b_returns = self.value_rms.normalize(b_returns)
                    b_values = self.value_rms.normalize(b_values)

                log_probs, entropy, values, mu, sigma = self.actor_critic.evaluate(b_obs, b_actions)

                log_ratio = log_probs - b_log_probs
                ratio = torch.exp(log_ratio)
                clip_0 = ratio * b_advantages
                clip_1 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_advantages

                L_clip = torch.min(clip_0, clip_1)

                if self.clip_value:
                    values_clipped = b_values + torch.clamp(values - b_values, -self.clip_eps, self.clip_eps)
                    L_vf = torch.max(torch.square(values - b_returns), torch.square(values_clipped - b_returns))
                else:
                    L_vf = torch.square(values - b_returns)

                loss = -1 * torch.mean(L_clip - self.c1_value * L_vf + self.c2_entropy * entropy)

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean() # low-variance KL estimator, logged for comparison
                    # exact KL(old || new) between diagonal Gaussians, summed over action dims (same as rsl_rl)
                    kl_exact = (
                        torch.log(sigma / b_sigma)
                        + (b_sigma.square() + (b_mu - mu).square()) / (2.0 * sigma.square())
                        - 0.5
                    ).sum(dim=-1).mean()

                # adaptive lr: adjust before this minibatch's step (same as rsl_rl)
                if self.desired_kl is not None:
                    kl = kl_exact.item()
                    lr = self.optimizer.param_groups[0]["lr"]
                    if kl > 2.0 * self.desired_kl:
                        lr = max(MIN_LR, lr / 1.5)
                    elif 0.0 < kl < self.desired_kl / 2.0:
                        lr = min(MAX_LR, lr * 1.5)
                    for group in self.optimizer.param_groups:
                        group["lr"] = lr

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm) # returns pre-clip norm
                self.optimizer.step()

                with torch.no_grad():
                    stats.add(
                        loss = loss,
                        policy_loss = -L_clip.mean(),
                        value_loss = L_vf.mean(),
                        entropy = entropy.mean(),
                        approx_kl = approx_kl,
                        kl = kl_exact,
                        clip_frac = ((ratio - 1).abs() > self.clip_eps).float().mean(),
                        ratio_max = ratio.max(),
                        grad_norm = grad_norm
                    )

        with torch.no_grad():
            _, std = self.actor_critic.get_mean_std(b_obs)
        return {**stats.means(), "action_std": std.mean().item(), "lr": self.optimizer.param_groups[0]["lr"]}



                



         