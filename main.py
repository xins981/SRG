from stable_baselines3 import SAC
from environment import Environment
from pointnet import PointNetExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import os, time, datetime
os.environ['DISPLAY'] = ':10.0'



# env = Environment(vis=True)

# policy_kwargs = dict(features_extractor_class=PointNetExtractor,
#                     features_extractor_kwargs=dict(features_dim=1088),
#                     net_arch=[128, 64])

# model = SAC(policy="MlpPolicy", env=env, learning_starts=0, batch_size=64, 
#             policy_kwargs=policy_kwargs, verbose=1)
# model.learn(total_timesteps=1000, log_interval=1)
# vec_env = model.get_env()
# obs, info = vec_env.reset()

# # obs, info = env.reset() # obs: (D, N)
# while True:
#     # Random action
#     # action_point_ind = np.random.randint(0, env.observation_space.shape[0])
#     # action_point = obs[:,action_point_ind]
#     # action_off = env.action_space.sample()
#     # action_off = action_off[3:] # drop random anchor coordinates
#     # action = np.concatenate((action_point, action_off))
#     # obs, reward, terminated, truncated, info = env.step(action)
#     # if terminated:
#     #     obs, info = env.reset()

#     action, state = model.predict(obs)
#     obs, reward, terminated, truncated, info = vec_env.step(action)
#     if terminated:
#         obs, info = vec_env.reset()



def make_env(rank, seed=0):
    
    def _init():
        env = Environment()
        env.reset(seed=seed + rank)
        return env
    
    set_random_seed(seed)

    return _init

if __name__ == "__main__":
    # n_training_envs = 64
    # train_env = SubprocVecEnv([make_env(rank=i) for i in range(n_training_envs)])
    
    # train_log_dir = f"logs/train/{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d.%H:%M:%S')}"
    # policy_kwargs = dict(features_extractor_class=PointNetExtractor, 
    #                      features_extractor_kwargs=dict(features_dim=1088), 
    #                      net_arch=[256, 128, 64])
    # model = SAC(env=train_env, policy="MlpPolicy", policy_kwargs=policy_kwargs, batch_size=64,
    #             gradient_steps=32, tensorboard_log=train_log_dir, verbose=1)
    # model.learn(total_timesteps=100_000)
    # model.save("sac_grasp")
    # model.save_replay_buffer("sac_replay_buffer")

    

    loaded_model = SAC.load("sac_grasp")
    # load it into the loaded_model
    # loaded_model.load_replay_buffer("sac_replay_buffer")

    # now the loaded replay is not empty anymore
    # print(f"The loaded_model has {loaded_model.replay_buffer.size()} transitions in its buffer")
    eval_env = Environment(vis=True)
    mean_reward, std_reward = evaluate_policy(loaded_model, eval_env, n_eval_episodes=10, deterministic=True, warn=False)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")