from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.sac.policies import Actor, MlpPolicy, SACPolicy

SelfSAC = TypeVar("SelfSAC", bound="SAC")


class SAC(OffPolicyAlgorithm):
    
    policy_aliases = {"MlpPolicy": MlpPolicy}
    policy: SACPolicy
    actor: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic
    
    def __init__(
        self,
        policy,
        env,
        learning_rate=3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        blm_update_step=5000,
        blm_end=0.1,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None

        self.blm_update_step = blm_update_step
        self.blm_end = blm_end
        
        if _init_setup_model:
            self._setup_model()

    
    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    
    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    
    def train(self, gradient_steps, batch_size=64):
        
        ''' based data in replay buffer to gradient aescent
        ''' 

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state; new_params:(B, N, 7), new_log_prob:(B, N).
            new_params, new_log_prob = self.actor.action_log_prob(replay_data.observations)
            
            if self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            with th.no_grad():
                # Select action according to policy; next_params:(B, N, 7), next_log_prob:(B, N).
                next_params, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q = th.cat(self.critic_target(replay_data.next_observations, next_params), dim=2).min(dim=2).values # (B, N)
                next_location_prob = th.nn.functional.softmax(next_q/self.policy.boltzmann_beta, dim=1) # (B, N)
                next_anchor_index = next_location_prob.multinomial(num_samples=1) # (B, 1)
                next_action_location_prob = th.gather(next_location_prob.unsqueeze(-1), 1, next_anchor_index.unsqueeze(-1).expand(-1, -1, 1)).squeeze() # (B, )
                next_action_log_prob = th.gather(next_log_prob.unsqueeze(-1), 1, next_anchor_index.unsqueeze(-1).expand(-1, -1, 1)).squeeze() # (B, )
                next_joint_log_prob = th.log(next_action_location_prob) + next_action_log_prob # (B, )
                next_action_q = th.gather(next_q.unsqueeze(-1), 1, next_anchor_index.unsqueeze(-1).expand(-1, -1, 1)).squeeze() # (B, )
                
                # add entropy term
                next_q_value_with_entropy = next_action_q - ent_coef * next_joint_log_prob # (B, )
                
                # td error + entropy term; rewards:(B, 1), dones:(B, 1).
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_value_with_entropy.unsqueeze(-1)

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            self.critic.requires_grad_(True)
            current_q = self.critic(replay_data.observations, replay_data.actions) # 2 *（B, 1)
            
            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(pred_q, target_q) for pred_q in current_q)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            self.critic.requires_grad_(False)
            new_q = th.cat(self.critic(replay_data.observations, new_params), dim=2).min(dim=2).values # (B, N)
            new_location_prob = th.nn.functional.softmax(new_q/self.policy.boltzmann_beta, dim=1) # (B, N)
            new_anchor_index = new_location_prob.multinomial(num_samples=1) # (B, 1)
            new_action_location_prob = th.gather(new_location_prob.unsqueeze(-1), 1, new_anchor_index.unsqueeze(-1).expand(-1, -1, 1)).squeeze() # (B, )
            new_action_log_prob = th.gather(new_log_prob.unsqueeze(-1), 1, new_anchor_index.unsqueeze(-1).expand(-1, -1, 1)).squeeze() # (B, )
            new_joint_log_prob = th.log(new_action_location_prob) + new_action_log_prob # (B, )
            new_action_q = th.gather(new_q.unsqueeze(-1), 1, new_anchor_index.unsqueeze(-1).expand(-1, -1, 1)).squeeze() # (B, )
            actor_loss = (ent_coef * new_joint_log_prob - new_action_q).mean()
            actor_losses.append(actor_loss.item())
           
            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef_loss = -(self.log_ent_coef * (new_joint_log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/boltzmann_coef", self.policy.boltzmann_beta)
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    
    def learn(self, total_timesteps, callback=None, log_interval=4, tb_log_name="SAC", reset_num_timesteps=True, progress_bar=False):
        
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
            blm_update_step=self.blm_update_step,
            blm_end=self.blm_end,
        )
    
    
    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]  # noqa: RUF005

    
    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables

