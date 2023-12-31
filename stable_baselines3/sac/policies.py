from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch as th
from gymnasium import spaces
from torch import nn
import numpy as np

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import Schedule
from utils import save_pcd, save_q_map

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Actor(BasePolicy):
    
    """ Actor network (policy) for SAC.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    action_space: spaces.Box

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch # actor middle layers, don't include input and output layer
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean
        action_dim = get_action_dim(self.action_space)
        latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn)
        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        if self.use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=last_layer_dim, log_std_init=log_std_init
            )
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)  # type: ignore[assignment]
            # actor output layer
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)  # type: ignore[assignment]
            self.score = nn.Linear(last_layer_dim, 1)
            self.sigmoid = nn.Sigmoid()


    def _get_constructor_parameters(self) -> Dict[str, Any]:
        
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                use_expln=self.use_expln,
                features_extractor=self.features_extractor,
                clip_mean=self.clip_mean,
            )
        )
        return data


    def get_std(self) -> th.Tensor:
        """
        Retrieve the standard deviation of the action distribution.
        Only useful when using gSDE.
        It corresponds to ``th.exp(log_std)`` in the normal case,
        but is slightly different when using ``expln`` function
        (cf StateDependentNoiseDistribution doc).

        :return:
        """
        msg = "get_std() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        return self.action_dist.get_std(self.log_std)


    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        msg = "reset_noise() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    
    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        
        features = self.extract_features(obs, self.features_extractor) # (B, N, 1088)
        latent_pi = self.latent_pi(features) # mlp (B, N, 256)
        pred_score = self.sigmoid(self.score(latent_pi)).squeeze() # (B, N)

        num_pts_objs = int(self.observation_space.shape[0] * 0.8)
        pts_objs = obs[:, :num_pts_objs, :] # (B, num_obj, 3)
        pred_score_objs = pred_score[:, :num_pts_objs] # (B, num_obj)
        latent_pi_objs = latent_pi[:, :num_pts_objs, :] # (B, num_obj, 1088)

        threshold_score = 0.5
        num_anchor = 64
        B, C = obs.shape[0], obs.shape[2]
        if B == 1:
            positive_pc_mask = (pred_score.view(-1) > threshold_score)
            positive_pc_mask = (pred_score.view(-1) > threshold_score)
            positive_pc_mask = positive_pc_mask.cpu().numpy()
            map_index = th.Tensor(np.nonzero(positive_pc_mask)[0]).view(-1).long()

            center_pc = th.full((num_anchor, C), -1.0)
            center_pc_index = th.full((num_anchor,), -1)

            pc = pc.view(-1,C)
            cur_pc = pc[map_index,:]
            if len(cur_pc) > num_anchor:
                center_pc_index[i] = th.Tensor(np.random.choice(cur_pc.shape[0], num_anchor, replace=False))
                # center_pc_index = _F.farthest_point_sample(cur_pc[:,:3].view(1,-1,3).transpose(2,1), num_anchor).view(-1)

                center_pc_index = map_index[center_pc_index.long()]
                center_pc = pc[center_pc_index.long()]
                
            elif len(cur_pc) > 0:
                center_pc_index[:len(cur_pc)] = th.arange(0, len(cur_pc))
                center_pc_index[len(cur_pc):] = th.Tensor(np.random.choice(cur_pc.shape[0], num_anchor-len(cur_pc), replace=True))
                center_pc_index = map_index[center_pc_index.long()]
                center_pc = pc[center_pc_index.long()]
                
            else:
                center_pc_index = th.Tensor(np.random.choice(pc.shape[0], num_anchor, replace=False))
                center_pc = pc[center_pc_index.long()]
        
            center_pc = center_pc.view(1,-1,C)
            center_pc_index = center_pc_index.view(1,-1)
        
        else:
            positive_mask = (pred_score_objs > threshold_score)
            anchor = th.full((B, num_anchor, C), -1.0)
            anchor_index_in_pcd = th.full((B, num_anchor), -1)
            anchor_feature = th.full((B, num_anchor, latent_pi_objs.shape[-1]), -1.0)
            anchor_score = th.full((B, num_anchor), -1.0)
            for i in range(B):
                positive_pts = pts_objs[i, positive_mask[i], :]

                if len(positive_pts) > num_anchor:
                    anchor_index_in_positive = th.Tensor(np.random.choice(positive_pts.shape[0], num_anchor, replace=False))
                    # center_pc_index[i] = farthest_point_sample(positive_pts, num_anchor)
                    # center_pc_index[i] = _F.farthest_point_sample(positive_pts[:,:3].view(1,-1,3).transpose(2,1), num_anchor).view(-1)

                    positive_index_in_pcd = th.nonzero(positive_mask[i]).view(-1)
                    index_in_pcd = positive_index_in_pcd[anchor_index_in_positive.long()]
                    anchor_index_in_pcd[i] = index_in_pcd
                    anchor[i] = pts_objs[i, index_in_pcd]
                    anchor_feature[i] = latent_pi_objs[i, index_in_pcd]
                    anchor_score[i] = pred_score_objs[i, index_in_pcd]
                    
                elif len(positive_pts) > 0:
                    positive_index = th.arange(0, len(positive_pts))
                    padding_index = th.Tensor(np.random.choice(positive_pts.shape[0], num_anchor-len(positive_pts), replace=True))
                    anchor_index_in_positive = th.cat((positive_index, padding_index))

                    positive_index_in_pcd = th.nonzero(positive_mask[i]).view(-1)
                    index_in_pcd = positive_index_in_pcd[anchor_index_in_positive.long()]
                    anchor_index_in_pcd[i] = index_in_pcd
                    anchor[i] = pts_objs[i, index_in_pcd]
                    anchor_feature[i] = latent_pi_objs[i, index_in_pcd]
                    anchor_score[i] = pred_score_objs[i, index_in_pcd]
                    
                else:
                    if pts_objs.shape[1] >= num_anchor:
                        index_in_pcd = th.Tensor(np.random.choice(pts_objs.shape[1], num_anchor, replace=False)).long()
                    else:
                        index_in_pcd = th.Tensor(np.random.choice(pts_objs.shape[1], num_anchor, replace=True)).long()
                    anchor_index_in_pcd[i] = index_in_pcd
                    anchor[i] = pts_objs[i, index_in_pcd]
                    anchor_feature[i] = latent_pi_objs[i, index_in_pcd]
                    anchor_score[i] = pred_score_objs[i, index_in_pcd]

        anchor = anchor.cuda() # (B, K, 3)
        anchor_score = anchor_score.cuda() # (B, K)
        anchor_feature = anchor_feature.cuda() # (B, K, 256)
        anchor_index_in_pcd = anchor_index_in_pcd.cuda() # (B, K) 

        mean_actions = self.mu(anchor_feature) # (B, K, 7)
        log_std = self.log_std(anchor_feature)  # (B, K, 7)
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        return mean_actions, log_std, anchor, anchor_index_in_pcd, anchor_score, {}
    
    
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        
        mean_actions, log_std, anchor, anchor_index_in_pcd, anchor_score, kwargs = self.get_action_dist_params(obs) # (B, N, 7)
        # Note: the action is squashed
        params = self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)
        norm_params = self.to_normal_param(anchor, params) # (B, K, 7)
        norm_params_with_id = th.cat([norm_params, anchor_index_in_pcd.unsqueeze(-1)], dim=-1)

        return norm_params_with_id, anchor_score

    
    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:

        mean_actions, log_std, anchor, anchor_index_in_pcd, anchor_score, kwargs = self.get_action_dist_params(obs) # (B, N, 7)
        # return action and associated log prob
        params, log_prob = self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs) # (B, K, 7) (B, K, 1)
        norm_params = self.to_normal_param(anchor, params)
        norm_params_with_id = th.cat([norm_params, anchor_index_in_pcd.unsqueeze(-1)], dim=-1)
        
        return norm_params_with_id, log_prob, anchor_score

    
    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self(observation, deterministic)


    def to_normal_param(self, obs, tanh_params):
        
        device = tanh_params.device
        low = th.tensor(self.action_space.low).to(device)
        high = th.tensor(self.action_space.high).to(device)
        untanh_params = low + (0.5 * (tanh_params + 1.0) * (high-low)) # (B, N, 7)
        axis_y_norm = th.norm(untanh_params[:,:,3:6], keepdim=True, dim=-1) # (B, N, 1)
        axis_y_unit = untanh_params[:,:,3:6] / axis_y_norm # (B, N, 3)
        tcp = untanh_params[:,:,:3] + obs[:,:,:3]
        rot_radian = untanh_params[:,:,-1].unsqueeze(-1) # (B, N, 1)
        normal_params = th.cat((tcp, axis_y_unit, rot_radian), dim=-1) # (B, N, 7)
        
        return normal_params



class SACPolicy(BasePolicy):
    
    """ Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    actor: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        boltzmann_beta=0.1,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [256, 256]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()

        sde_kwargs = {
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)
        
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.share_features_extractor = share_features_extractor
        self.boltzmann_beta = boltzmann_beta
        self._build(lr_schedule)


    def _build(self, lr_schedule: Schedule) -> None:
        
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Do not optimize the shared features extractor with the critic loss
            # otherwise, there are gradient computation issues
            critic_parameters = [param for name, param in self.critic.named_parameters() if "features_extractor" not in name]
        else:
            # Create a separate features extractor for the critic
            # this requires more memory and computation
            self.critic = self.make_critic(features_extractor=None)
            critic_parameters = list(self.critic.parameters())

        # Critic target should not share the features extractor with critic
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic.optimizer = self.optimizer_class(
            critic_parameters,
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)


    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                use_sde=self.actor_kwargs["use_sde"],
                log_std_init=self.actor_kwargs["log_std_init"],
                use_expln=self.actor_kwargs["use_expln"],
                clip_mean=self.actor_kwargs["clip_mean"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data


    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        self.actor.reset_noise(batch_size=batch_size)


    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor(**actor_kwargs).to(self.device)


    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)


    def forward(self, obs, deterministic, data_dir, rollout):
        
        params_pointwise, center_pc_score = self.actor(obs, deterministic=deterministic) # (B, K, 8)
        # num_objs = int(self.observation_space.shape[0] * 0.8)
        # q_values = th.cat(self.critic(obs, params_pointwise, center_pc_index), dim=2) # (B, K, 2)
        # min_q_values = th.min(q_values, dim=-1).values # (B, K)
        # obj_min_q_values = min_q_values[:,:num_objs] # (B, K)
        score_distribution = th.nn.functional.softmax(center_pc_score/self.boltzmann_beta, dim=-1) # (B, K)
        if deterministic == True:
            anchor_index = th.max(score_distribution, dim=1, keepdim=True).indices # (B, 1)
        else:
            anchor_index = score_distribution.multinomial(num_samples=1) # (B, 1)
        
        # if data_dir != None and (rollout - 1) % 20 == 0:
        #     pts = obs[:,:,:3].cpu().numpy() # (B, N, 3)
        #     q_value = min_q_values.cpu().numpy() # (B, N)
        #     save_pcd(pts=pts, data_dir=data_dir, rollout=rollout)
        #     save_q_map(pts=pts, q_value=q_value, data_dir=data_dir, rollout=rollout)

        action_param = th.gather(params_pointwise, 1, anchor_index.unsqueeze(-1).expand(-1, -1, params_pointwise.shape[-1])).squeeze(1) # (B, 8)
        
        return action_param


    def _predict(self, observation, deterministic=False, data_dir=None, rollout=None):
        return self(observation, deterministic, data_dir, rollout)


    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode



MlpPolicy = SACPolicy



class CnnPolicy(SACPolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )



class MultiInputPolicy(SACPolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )
