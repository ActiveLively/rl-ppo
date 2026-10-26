import torch

def calculate_gae(rewards, values, next_values, terminated, truncated, gamma, lam):
    '''
    traditional GAE logic:
    δ_t = r_t + γ·V(s_{t+1})·(1 - dones) - V(s_t)  
    Â_t = δ_t + γλ·(1-dones)·Â_{t+1}.

    new GAE logic to account for terminated (failed) and truncated (run wout of time) ends
    '''
    num_steps, num_envs = rewards.shape # (T steps, N envs)s

    not_dones = 1 - torch.maximum(terminated, truncated) 
    values = torch.cat((values, next_values.unsqueeze(0)), dim=0) # [T,N] + [1, N] = [T+1, N]

    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[0]) # puts gae on the same device and shape as next_value (N, )

    # values[:-1] b/c of values = torch.cat the next_values above
    not_terminateds = 1 - terminated
    truncated_only = (truncated * not_terminateds)
    rewards = rewards + gamma * values[:-1] * truncated_only

    for t in reversed(range(num_steps)):
        delta = rewards[t, :] + gamma * values[t+1, :] * not_dones[t] - values[t, :]
        gae = delta + gamma * lam * not_dones[t] * gae
        advantages[t, :] = gae

    returns = advantages + values[:-1] # G_t (bootstraped est. return) = A_t (expected) + V_t (critic expected value)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


class Buffer():
    def __init__(self, num_steps, num_envs, obs_dim, action_dim, device, gamma, lam):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.lam = lam

        self.obs = torch.zeros(self.num_steps, self.num_envs, self.obs_dim, device=self.device)
        self.actions = torch.zeros(self.num_steps, self.num_envs, self.action_dim, device=self.device)
        self.log_prob = torch.zeros(self.num_steps, self.num_envs, device=self.device)
        self.rewards = torch.zeros(self.num_steps, self.num_envs, device=self.device)
        self.terminateds = torch.zeros(self.num_steps, self.num_envs, device=self.device)
        self.truncateds = torch.zeros(self.num_steps, self.num_envs, device=self.device)
        self.values = torch.zeros(self.num_steps, self.num_envs, device=self.device)
        self.advantages = torch.zeros(self.num_steps, self.num_envs, device=self.device)
        self.returns = torch.zeros(self.num_steps, self.num_envs, device=self.device)

        self.step = 0

    def reset(self):
        self.step = 0

    def add(self, obs, action, log_prob, reward, terminated, truncated, value):
        assert self.step < self.num_steps # should not trigger

        self.obs[self.step] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[self.step] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.log_prob[self.step] = torch.as_tensor(log_prob, dtype=torch.float32, device=self.device)
        self.rewards[self.step] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.terminateds[self.step] = torch.as_tensor(terminated, dtype=torch.float32, device=self.device)
        self.truncateds[self.step] = torch.as_tensor(truncated, dtype=torch.float32, device=self.device)
        self.values[self.step] = torch.as_tensor(value, dtype=torch.float32, device=self.device)

        self.step += 1

    @torch.no_grad()
    def compute_advantage_return(self, next_values):
        assert self.step == self.num_steps # should not trigger

        next_values = torch.as_tensor(next_values, dtype=torch.float32, device=self.device)
        self.advantages, self.returns = calculate_gae(
            self.rewards, self.values, next_values, self.terminateds, self.truncateds, self.gamma, self.lam
            )

    def get_minibatches(self, minibatch_size):
        batch_size = self.num_steps * self.num_envs

        b_obs = self.obs.reshape(batch_size, self.obs_dim)
        b_actions = self.actions.reshape(batch_size, self.action_dim)
        b_log_probs = self.log_prob.reshape(batch_size)
        b_advantages = self.advantages.reshape(batch_size)
        b_returns = self.returns.reshape(batch_size)
        b_values = self.values.reshape(batch_size)

        indices = torch.randperm(batch_size, device=self.device)

        for start in range(0,batch_size, minibatch_size):
            mb = indices[start:start + minibatch_size]

            yield {
                "obs": b_obs[mb],
                "actions": b_actions[mb],
                "log_probs": b_log_probs[mb],
                "advantages": b_advantages[mb],
                "returns": b_returns[mb],
                "values": b_values[mb]
            }
