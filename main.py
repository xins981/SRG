from stable_baselines3 import SAC
from environment import Environment
from pointnet import PointNetExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, HParamCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import os, datetime
import yaml
os.environ['DISPLAY'] = ':10.0'
import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id='PandaGrasp-v0',
    entry_point='environment:Environment',
    kwargs={'reward_scale': -1, 'vis': False},
)

def make_env(rank, seed=0):
    
    def _init():
        
        env = gym.make('PandaGrasp-v0', max_episode_steps=50)
        # env = Monitor(env)
        env.reset(seed=seed + rank)
        
        return env
    
    set_random_seed(seed)

    return _init


def train():
    
    experiment_dir = f'logs/experiment/{datetime.datetime.now().strftime("%Y-%m-%d.%H:%M:%S")}'
    policy_kwargs = dict(features_extractor_class=PointNetExtractor, 
                        features_extractor_kwargs=dict(features_dim=1088),
                        net_arch=[256, 256],
                        boltzmann_beta=5)
    
    n_training_envs = 24
    # n_eval_envs = 8
    total_timesteps = 1_000_000
    train_env = SubprocVecEnv([make_env(i) for i in range(n_training_envs)])
    # eval_env = SubprocVecEnv([make_env(rank=i, max_episode_len=max_episode_len, reward_scale=reward_scale) for i in range(n_training_envs, n_training_envs + n_eval_envs)])

    for i in range(1, 2): # different hyper-parameters

        for j in range(1, 2): # different seed

            session_dir = f'{experiment_dir}/trial_{i}/session_{j}'
            
            # checkpoint_callback = CheckpointCallback(save_freq=max(100000 // n_training_envs, 1), save_path=f'{session_dir}/checkpoints', 
            #                                          name_prefix='sac_model', save_replay_buffer=True)
            # eval_callback = EvalCallback(eval_env, best_model_save_path=session_dir, log_path=session_dir, 
            #                              deterministic=True, eval_freq=max(5000 // n_training_envs, 1), n_eval_episodes=8)
            # callback_list = CallbackList([checkpoint_callback, eval_callback])

            model = SAC('MlpPolicy', train_env, policy_kwargs=policy_kwargs, 
                        batch_size=64, gradient_steps=1, learning_starts=1000, 
                        blm_update_step=20000, blm_end=0.1, tensorboard_log=session_dir, verbose=1)
            # model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
            model.learn(total_timesteps=total_timesteps, progress_bar=True)
            
            with open(f'{session_dir}/hyparam.yaml','a') as f:
                hyparam_dict = {
                    'max episode step': 50,
                    'reward scale': -1,
                    'total timesteps': total_timesteps,
                    'env number': model.n_envs,
                    'buffer size': model.buffer_size,
                    'learning start': f'{model.learning_starts} step',
                    'train frequency': f'{model.train_freq.frequency} rollout',
                    'learning rate': model.learning_rate,
                    'gamma': model.gamma,
                    'tau': model.tau,
                    'gradient step': model.gradient_steps,
                    'batch size': model.batch_size,
                    'feature dim': model.policy_kwargs['features_extractor_kwargs']['features_dim'],
                    'net arch': str(model.policy_kwargs['net_arch']),
                }
                yaml.dump(hyparam_dict, f)

            model.save(f'{session_dir}/trained_model')
            model.save_replay_buffer(f'{session_dir}/trained_model_replay_buffer')
    
    train_env.close()
    # eval_env.close()


def eval(model_dir, log_dir):
    best_model = SAC.load(model_dir)
    eval_env = Environment(vis=True)
    eval_env = Monitor(eval_env, filename=log_dir)
    mean_reward, std_reward = evaluate_policy(best_model, eval_env, n_eval_episodes=5, deterministic=True)
    print(f'mean_reward={mean_reward:.2f} +/- {std_reward}')


if __name__ == '__main__':
    
    # train()

    loaded_model = SAC.load('logs/experiment/2023-08-16.11:03:11/trial_1/session_1/trained_model.zip', verbose=1)
    loaded_model.set_env(SubprocVecEnv([make_env(i) for i in range(24)]))
    loaded_model.load_replay_buffer('logs/experiment/2023-08-16.11:03:11/trial_1/session_1/trained_model_replay_buffer.pkl')
    loaded_model.set_parameters('logs/experiment/2023-08-16.11:03:11/trial_1/session_1/trained_model.zip')
    loaded_model.learn(1_000_000, reset_num_timesteps=False, progress_bar=True)

    # eval(model_dir='logs/experiment/2023-07-27.12:13:10/trial_1/session_1/best_model.zip', 
    #      log_dir='logs/experiment/2023-07-24.22:16:04/trial_1/session_1/eval')

    