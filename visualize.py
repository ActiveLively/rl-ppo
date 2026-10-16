import torch
import gymnasium as gym
import gymnasium_robotics
from actor_critic import ActorCritic

gym.register_envs(gymnasium_robotics)

ckpt = torch.load("checkpoints/fetch_reach.pt", weights_only=False)

env = gym.make("FetchReachDense-v4", render_mode="human")
env = gym.wrappers.FlattenObservation(env)
env = gym.wrappers.NormalizeObservation(env)
env.update_running_mean = False # freeze stats, use the ones from training
env.obs_rms.mean = ckpt["obs_mean"]
env.obs_rms.var = ckpt["obs_var"]

# must match the architecture in main.py
actor_critic = ActorCritic(
    input_dim = env.observation_space.shape[0],
    actor_hidden_dim = [256, 256],
    critic_hidden_dim = [256, 256],
    action_dim = env.action_space.shape[0],
    activation = "tanh",
    norm = "layer",
    state_dependent = True,
)
actor_critic.load_state_dict(ckpt["model"])
actor_critic.eval()

for ep in range(10):
    obs, _ = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            mean, _ = actor_critic.get_mean_std(torch.as_tensor(obs, dtype=torch.float32))
        action = mean.clamp(-1, 1).numpy() # deterministic: mean action, no exploration noise
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    print(f"episode {ep}: success={info['is_success']}")

env.close()
