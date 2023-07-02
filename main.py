import os
from environment import Environment
import numpy as np



os.environ['DISPLAY'] = ':10.0'


env = Environment()

# policy_kwargs = dict(
#     features_extractor_class=PointNetExtractor,
#     features_extractor_kwargs=dict(features_dim=1024),
#     net_arch=[512, 256, 128],
# )

# model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
# model.learn(1000)


obs, info = env.reset()
n_steps = 100
for _ in range(n_steps):
    # Random action
    action_point_ind = np.random.randint(0, env.observation_space.shape[0])
    action_point = obs[action_point_ind,:]
    action_off = env.action_space.sample()
    action_off = action_off[3:] # drop random anchor coordinates
    action = np.concatenate((action_point, action_off))
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        obs, info = env.reset()