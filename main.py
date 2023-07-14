from stable_baselines3 import SAC
from environment import Environment
from pointnet import PointNetExtractor
import os
import numpy as np



os.environ['DISPLAY'] = ':11.0'


env = Environment(vis=True)

policy_kwargs = dict(features_extractor_class=PointNetExtractor,
                    features_extractor_kwargs=dict(features_dim=1088),
                    net_arch=[128, 128])

model = SAC(policy="MlpPolicy", env=env, learning_starts=0, batch_size=64, 
            policy_kwargs=policy_kwargs, verbose=1, learning_rate=0.0001)
model.learn(total_timesteps=1000, log_interval=1)
vec_env = model.get_env()
obs, info = vec_env.reset()

# obs, info = env.reset() # obs: (D, N)
while True:
    # Random action
    # action_point_ind = np.random.randint(0, env.observation_space.shape[0])
    # action_point = obs[:,action_point_ind]
    # action_off = env.action_space.sample()
    # action_off = action_off[3:] # drop random anchor coordinates
    # action = np.concatenate((action_point, action_off))
    # obs, reward, terminated, truncated, info = env.step(action)
    # if terminated:
    #     obs, info = env.reset()

    action, state = model.predict(obs)
    obs, reward, terminated, truncated, info = vec_env.step(action)
    if terminated:
        obs, info = vec_env.reset()