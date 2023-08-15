import gymnasium as gym
import numpy as np
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
    anchor = observation[anchor_ind, :3]
    # anchor = np.mean(observation[:int(env.observation_space.shape[0] * 0.8), :], axis=0)
    unscaled_params = env.action_space.sample()
    axis_y_norm = np.linalg.norm(unscaled_params[3:6])
    unscaled_params[:3] = (unscaled_params[:3] * 0.05) + anchor
    unscaled_params[3:6] /= axis_y_norm
    unscaled_params[6] = (0.5 * (unscaled_params[6] + 1.0) * np.pi)

    observation, reward, terminated, truncated, info = env.step(unscaled_params)

    if terminated or truncated:
        observation, info = env.reset()

env.close()