import gymnasium as gym
import numpy as np
# from environment import Environment
from gymnasium.envs.registration import register

register(
    id='PandaGrasp-v0',
    entry_point='environment:Environment',
    kwargs={'reward_scale': 10, 'vis': True},
)

env = gym.make('PandaGrasp-v0', max_episode_steps=50)

observation, info = env.reset()
for _ in range(1000):
    as_high = env.action_space.high
    as_low = env.action_space.low
    anchor_ind = np.array([np.random.randint(0, int(env.observation_space.shape[0] * 0.8))])
    anchor = observation[anchor_ind, :]
    # anchor = np.mean(observation[:int(env.observation_space.shape[0] * 0.8), :], axis=0)
    params = env.action_space.sample()
    action = as_low + (0.5 * (params + 1.0) * (as_high - as_low)) # (7, )
    axis_y_norm = np.linalg.norm(action[3:6])
    action[3:6] /= axis_y_norm
    action[:3] = (action[:3] * 0.02) + anchor

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()