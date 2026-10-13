import torch 
from actor_critic import ActorCritic
from buffer import Buffer


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
        ):
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

        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr = self.lr)

    @torch.no_grad()
    def rollout(self):
        self.buffer.reset()
        obs = self.env.get_obs()
        
        for i in range(self.steps):
            action, log_prob, value = self.actor_critic.act(obs)
            next_obs, reward, done = self.env.step(action)
            self.buffer.add(obs, action, log_prob, reward, done, value)

            obs = next_obs

        # bootstrap value of the state after the last collected step -> [num_envs]
        next_values = self.actor_critic.value(obs)

        self.buffer.compute_advantage_return(next_values)


    def update(self):
        for epoch in range(self.epochs):
            for mb in self.buffer.get_minibatches(minibatch_size = self.minibatch_size):
                b_obs = mb["obs"]
                b_actions = mb["actions"]
                b_log_probs = mb["log_probs"]
                b_advantages = mb["advantages"]
                b_returns = mb["returns"]
                b_values = mb["values"]

                log_probs, entropy, values = self.actor_critic.evaluate(b_obs, b_actions)

                ratio = torch.exp(log_probs - b_log_probs)
                clip_0 = ratio * b_advantages
                clip_1 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_advantages

                L_clip = torch.min(clip_0, clip_1)

                L_vf = torch.square(values - b_returns)

                loss = -1 * torch.mean(L_clip - self.c1_value * L_vf + self.c2_entropy * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5) # suggested for PPO gradient clipping
                self.optimizer.step()



                



         