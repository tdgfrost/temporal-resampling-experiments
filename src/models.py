from typing import Tuple, Dict, Optional
import numpy as np

import gymnasium as gym
from stable_baselines3 import PPO
import torch.nn as nn
import torch
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.distributions import Categorical
from tqdm import tqdm
from collections import deque
import random


def set_seed(seed: int):
    """
    Sets the random seed for reproducibility across torch, numpy, and random.
    """
    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)

    # Set seed for PyTorch on CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Set deterministic algorithms for PyTorch (can impact performance)
    # This is crucial for full reproducibility with CUDA
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # You might also want to set this for newer PyTorch versions
    # torch.use_deterministic_algorithms(True)

    # Set environment variable for CUDA (if needed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


class CallablePPO(PPO):
    """
    A version of PPO that can be called like a function to get actions.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, obs):
        action, _ = self.predict(obs, deterministic=True)
        return action


class MiniGridCNN(nn.Module):
    def __init__(self, C: int, H: int, W: int, feature_size: int = 128, dropout_p: float = 0.0):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=2),
            nn.GroupNorm(1, 16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=2),
            nn.GroupNorm(1, 32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=2),
            nn.GroupNorm(1, 64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, C, H, W)).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flat, feature_size),
            nn.LayerNorm(feature_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )

        self.scale_inputs_maybe = lambda x: x
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for Conv2d/Linear with Kaiming normal,
        biases with zeros, Norm layers with weight=1, bias=0.
        The final Linear layer in the decoder is zero-initialized.
        """
        def _init_fn(m: nn.Module):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # apply to all submodules
        self.apply(_init_fn)

    def _get_obs_shape(self, observation_shape: Tuple[int, int, int]):
        if observation_shape[-1] in (1, 5):
            H, W, C = observation_shape
            self.permute_obs_maybe = lambda x: x.permute(0, 3, 1, 2)
        else:
            C, H, W = observation_shape
            self.permute_obs_maybe = lambda x: x

        return C, H, W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        permuted_x = self.permute_obs_maybe(x)
        scaled_x = self.scale_inputs_maybe(permuted_x)
        shrink_x = self.shrink_obs(scaled_x)
        return self.fc(self.cnn(shrink_x))



class PPOMiniGridCNN(MiniGridCNN):
    def __init__(self, observation_shape: Tuple[int, int, int], feature_size: int = 128,
                 *args, **kwargs) -> None:
        C, H, W = self._get_obs_shape(observation_shape)

        C -= 1  # we will ignore the final channel (the always-0 channel in LavaGap)
        self.shrink_obs = lambda x: x[:, :C, :, :]

        super().__init__(C, H, W, feature_size)


class OfflineMiniGridCNN(MiniGridCNN):
    def __init__(self, observation_shape: Tuple[int, int, int], feature_size: int = 128,
                 input_scaling: bool = True, *args, **kwargs) -> None:

        C, H, W = self._get_obs_shape(observation_shape)

        C -= 1  # we will ignore the final channel (the always-0 channel in LavaGap)
        self.shrink_obs = lambda x: x[:, :C, :, :]

        super().__init__(C, H, W, feature_size)

        if input_scaling:
            self.scale_inputs_maybe = self.scale_inputs

    @staticmethod
    def scale_inputs(x):
        scaled_x = x.clone()
        scaled_x[:, 1, :, :] /= 3.0
        scaled_x[:, 2, :, :] /= 10.0
        scaled_x[:, 3, :, :] /= 5.0
        return scaled_x


class PPOMiniGridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512) -> None: #, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        self.net = PPOMiniGridCNN(observation_space.shape, feature_size=features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class CustomNet(nn.Module):
    def __init__(self, observation_shape: Tuple[int, int, int], output_size: int = 1, feature_size: int = 128,
                 device: str = 'cpu', feature_extractor=None, dropout_p: float = 0.0, *args, **kwargs) -> None:
        super().__init__()
        self._device = device
        if feature_extractor is None:
            self.encoder = OfflineMiniGridCNN(observation_shape=observation_shape,
                                              feature_size=feature_size,
                                              dropout_p=dropout_p).to(device)
        else:
            self.encoder = feature_extractor

        self.decoder = nn.Sequential(
            nn.Linear(feature_size, feature_size // 2),
            nn.LayerNorm(feature_size // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            nn.Linear(feature_size // 2, feature_size // 2),
            nn.LayerNorm(feature_size // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            nn.Linear(feature_size // 2, output_size)
        ).to(device)

        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for Conv2d/Linear with Kaiming normal,
        biases with zeros, Norm layers with weight=1, bias=0.
        The final Linear layer in the decoder is zero-initialized.
        """
        def _init_fn(m: nn.Module):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # apply to all submodules
        self.apply(_init_fn)

        # special-case: zero-init the very last decoder layer
        if isinstance(self.decoder[-1], nn.Linear):
            nn.init.zeros_(self.decoder[-1].weight)
            nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, x: torch.Tensor, actions: torch.Tensor = None) -> torch.Tensor:
        x, actions = self._ndarray_to_tensor(x, actions)
        x = x.to(dtype=torch.float32)

        # (N, C, H, W) -> (N, feature_size)
        hidden = self.encoder(x)

        # (N, feature_size) -> (N, output_size)
        output = self.decoder(hidden)

        if actions is None:
            return output

        # (N, output_size) -> (N, 1)
        return output.gather(1, actions.long())

    def _ndarray_to_tensor(self, *arrays):
        new_tensors = []
        for array in arrays:
            if array is None:
                new_tensors.append(None)
                continue
            if not isinstance(array, torch.Tensor):
                array = torch.tensor(array)
            new_tensors.append(array.to(self._device))

        return new_tensors

    def enable_mc_dropout(self):
        """Enable MC Dropout during inference."""
        def apply_mc_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()
        self.apply(apply_mc_dropout)


class _CustomBase(nn.Module):
    def __init__(self, device: str = 'cpu', seed: Optional[int] = None, *args, **kwargs):
        super().__init__()
        self._cloning_only = False
        self._device = device
        self.scaler = None
        self._scaler_dtype = None
        if seed is not None:
            self.seed = int(seed)
            set_seed(seed)

    def fit(
        self,
        dataset,
        epochs: int = 1,
        n_steps_per_epoch: int = 1_000,
        evaluators=None,
        show_progress: bool = True,
        experiment_name: str = None,
        dataset_kwargs: Optional[Dict] = None
    ):
        # Initialise our dataset and loss dictionary
        dataset_kwargs = dict() if dataset_kwargs is None else dataset_kwargs
        dataset.set_to_tensors(self._device)

        loss_dict = self._reset_loss_dict()

        if torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler('cuda')
            self._scaler_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        # Start training
        with tqdm(total=epochs * n_steps_per_epoch, desc="Progress", mininterval=2.0, disable=not show_progress) as pbar:
            for epoch in range(1, epochs + 1):
                epoch_str = f"{epoch}/{epochs}"

                for update_step in range(n_steps_per_epoch):
                    obs, acts, rews, next_obs, dones, flags, next_flags = dataset.sample_transition_batch(
                        self._batch_size, **dataset_kwargs
                    )

                    # Update the networks
                    if not self._cloning_only:
                        loss_dict['critic_loss'].append(self._update_critic(obs, acts, rews, next_obs, dones, flags, next_flags))
                        loss_dict['value_loss'].append(self._update_value(obs, acts))
                    loss_dict['policy_loss'].append(self._update_actor(obs, acts))

                    # Soft update of target value network
                    self.sync_target_networks()

                    if self.scaler is not None:
                        self.scaler.update()

                    pbar.update(1)
                    pbar.set_postfix(epoch=epoch_str,
                                     policy_loss=f"{np.mean(loss_dict['policy_loss']):.5f}",
                                     critic_loss=f"{np.mean(loss_dict['critic_loss']):.5f}",
                                     value_loss=f"{np.mean(loss_dict['value_loss']):.5f}",
                                     refresh=False)

                # Logging
                loss_dict, log_dict = self._log_progress(
                    epoch=epoch,
                    loss_dict=loss_dict,
                    experiment_name=experiment_name,
                    evaluators=evaluators
                )

        return log_dict

    def generic_update(self, loss, optimizer):
        optimizer.zero_grad()
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
        else:
            loss.backward()
            optimizer.step()
        return loss.detach().clone()

    def _update_critic(self, *args):
        return np.nan

    def _update_value(self, *args):
        return np.nan

    def _update_actor(self, *args):
        return np.nan

    def sync_target_networks(self):
        pass

    def _extract_flag(self, obs):
        obs = self._to_tensors(obs)[0]
        if obs.shape[-1] in (1, 5):
            return obs[:, -1, -1, 0].unsqueeze(-1).long()
        return obs[:, 0, -1, -1].unsqueeze(-1).long()

    def _to_tensors(self, *arrays):
        new_tensors = []
        for arr in arrays:
            if not isinstance(arr, torch.Tensor):
                arr = torch.tensor(arr, dtype=torch.float32)
            new_tensors.append(arr.to(self._device))
        return new_tensors

    def _log_progress(self, epoch: int, loss_dict: dict, experiment_name: str,
                      evaluators: dict = None):

        rewards = {}
        for key in evaluators.keys():
            mean_rew, std_rew = evaluators[key](self, seed=self.seed)
            rewards[key] = (mean_rew, std_rew)

        eval_str = '\n' + '=' * 40 + f"\nEpoch {epoch}:"
        for key in rewards.keys():
            eval_str += f"\n     {key} = {rewards[key][0]:.2f} +/- {rewards[key][1]:.2f}"

        eval_str += f"\n\n     policy_loss = {np.mean(loss_dict['policy_loss']):.7f}"
        eval_str += f"\n     critic_loss = {np.mean(loss_dict['critic_loss']):.7f}"
        eval_str += f"\n     value_loss = {np.mean(loss_dict['value_loss']):.7f}\n"
        eval_str += '=' * 40 + '\n'
        print(eval_str)

        return self._reset_loss_dict(), rewards

    @staticmethod
    def _reset_loss_dict():
        return {
            'critic_loss': deque(maxlen=100),
            'value_loss': deque(maxlen=100),
            'policy_loss': deque(maxlen=100)
        }


class CustomIQL(_CustomBase):
    def __init__(
            self,
            observation_shape: Tuple[int, int, int],
            action_size: int,
            input_length: int = 2,
            feature_size: int = 128,
            batch_size: int = 128,
            expectile: float = 0.7,
            gamma: float = 0.99,
            critic_lr: float = 3e-4,
            value_lr: float = 3e-4,
            actor_lr: float = 3e-4,
            tau_target: float = 0.005,
            dropout_p: float = 0.0,
            beta: float = 2.0,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._feature_size = feature_size
        self._batch_size = batch_size
        self._expectile = expectile
        self._gamma = gamma
        self._batch_diff = None
        self._input_length = input_length
        self._cloning_only = expectile == 0.5
        self._tau_target = tau_target
        self._has_dropout = dropout_p > 0.0
        self._beta = beta

        net_kwargs = dict(observation_shape=observation_shape, feature_size=feature_size, dropout_p=dropout_p, device=self._device)

        self.feature_extractor = OfflineMiniGridCNN(**net_kwargs).to(self._device)

        self.critic_net1 = CustomNet(output_size=action_size, feature_extractor=self.feature_extractor, **net_kwargs)
        self.critic_net2 = CustomNet(output_size=action_size, feature_extractor=self.feature_extractor, **net_kwargs)

        self.value_net = CustomNet(output_size=1, feature_extractor=self.feature_extractor, **net_kwargs)
        self.target_value_net = CustomNet(output_size=1, feature_extractor=self.feature_extractor, **net_kwargs)

        self.policy_net = CustomNet(output_size=action_size, feature_extractor=self.feature_extractor, **net_kwargs)

        # Give both critic nets to the critic optimizer
        self.critic_optim = torch.optim.AdamW(list(self.critic_net1.encoder.parameters()) +
                                             list(self.critic_net1.decoder.parameters()) +
                                             list(self.critic_net2.decoder.parameters()), lr=critic_lr)
        self.value_optim = torch.optim.AdamW(self.value_net.parameters(), lr=value_lr)
        self.policy_optim = torch.optim.AdamW(self.policy_net.parameters(), lr=actor_lr)

        # clone the value net to a target network
        self.sync_target_networks(tau=1.0)

    def forward(self, obs, acts):
        q1, q2 = self.critic_net1(obs, acts), self.critic_net2(obs, acts)
        v = self.value_net(obs)
        logits = self.policy_net(obs)
        return (q1, q2), v, logits

    def predict(self, obs, deterministic: bool = False):
        obs = self._to_tensors(obs)[0]
        logits = self.policy_net(obs)
        if deterministic:
            return logits.argmax(dim=-1).cpu().numpy()
        return Categorical(logits=logits).sample().cpu().numpy()

    def sync_target_networks(self, tau=None):
        tau = tau or self._tau_target
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)

    def _update_critic(self, *args):
        critic_loss = self._update_critic_compiled(*args)
        critic_loss = self.generic_update(critic_loss, self.critic_optim)
        return critic_loss.item()

    @torch.compile
    def _update_critic_compiled(self, obs, acts, rews, next_obs, dones, flags, next_flags):
        with torch.autocast(device_type="cuda", enabled=self.scaler is not None, dtype=self._scaler_dtype):
            q1, q2 = self.critic_net1(obs, acts), self.critic_net2(obs, acts)
            with torch.no_grad():
                next_multistep_discount = torch.where(next_flags == 1, 3, 1)
                current_multistep_discount = torch.where(flags == 1, 2, 0)

                v_next = self.target_value_net(next_obs)
                r = self._gamma ** current_multistep_discount * rews.float()
                q_target = r + self._gamma ** next_multistep_discount * (1 - dones.float()) * v_next

            critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        return critic_loss

    def _update_value(self, *args):
        value_loss = self._update_value_compiled(*args)
        value_loss = self.generic_update(value_loss, self.value_optim)
        return value_loss.item()

    @torch.compile
    def _update_value_compiled(self, obs, acts):
        with torch.autocast(device_type="cuda", enabled=self.scaler is not None, dtype=self._scaler_dtype):
            v = self.value_net(obs)
            with torch.no_grad():
                q1, q2 = self.critic_net1(obs, acts), self.critic_net2(obs, acts)
                q = torch.min(q1, q2)

            diff = q - v
            self._batch_diff = diff.detach().squeeze()
            weights = torch.absolute(self._expectile - (self._batch_diff < 0).float()).squeeze()
            value_loss = (weights * (diff.squeeze() ** 2)).mean()
        return value_loss

    def _update_actor(self, *args):
        policy_loss = self._update_actor_compiled(*args)
        policy_loss = self.generic_update(policy_loss, self.policy_optim)
        return policy_loss.item()

    @torch.compile
    def _update_actor_compiled(self, obs, acts):
        with torch.autocast(device_type="cuda", enabled=self.scaler is not None, dtype=self._scaler_dtype):
            logits = self.policy_net(obs)
            weights = 1.0
            if not self._cloning_only:
                weights = self._batch_diff
                if self._has_dropout:
                    weights = torch.absolute(self._expectile - (self._batch_diff < 0).float()).squeeze()
                else:
                    weights = torch.clip(torch.exp(self._beta * weights), -torch.inf, 100)

            policy_loss = F.cross_entropy(logits, acts.squeeze().long(), reduction='none')
            policy_loss = (policy_loss * weights).mean()
        return policy_loss


class CustomCQLSAC(_CustomBase):
    def __init__(
        self,
        observation_shape: tuple,
        action_size: int,
        feature_size: int = 128,
        batch_size: int = 128,
        gamma: float = 0.99,
        cql_alpha: float = 1.0,
        critic_lr: float = 3e-4,
        actor_lr: float = 3e-4,
        tau_target: float = 0.005,
        entropy_alpha: float = 0.2,
        dropout_p: float = 0.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._feature_size = feature_size
        self._batch_size = batch_size
        self._gamma = gamma
        self._tau_target = tau_target
        self._cql_alpha = cql_alpha
        self._entropy_alpha = entropy_alpha

        net_kwargs = dict(observation_shape=observation_shape, feature_size=feature_size, dropout_p=dropout_p, device=self._device)

        self.feature_extractor = OfflineMiniGridCNN(**net_kwargs).to(self._device)

        self.critic_net1 = CustomNet(output_size=action_size, feature_extractor=self.feature_extractor, **net_kwargs)
        self.critic_net2 = CustomNet(output_size=action_size, feature_extractor=self.feature_extractor, **net_kwargs)
        self.target_critic_net1 = CustomNet(output_size=action_size, feature_extractor=self.feature_extractor, **net_kwargs)
        self.target_critic_net2 = CustomNet(output_size=action_size, feature_extractor=self.feature_extractor, **net_kwargs)
        self.policy_net = CustomNet(output_size=action_size, feature_extractor=self.feature_extractor, **net_kwargs)

        self.critic_optim = torch.optim.AdamW(
            list(self.critic_net1.parameters()) + list(self.critic_net2.parameters()), lr=critic_lr
        )
        self.policy_optim = torch.optim.AdamW(self.policy_net.parameters(), lr=actor_lr)

        # Initialize target networks
        self.sync_target_networks(tau=1.0)

    def forward(self, obs):
        q1_logits = self.critic_net1(obs)
        q2_logits = self.critic_net2(obs)
        policy_logits = self.policy_net(obs)
        return q1_logits, q2_logits, policy_logits

    def predict(self, obs, deterministic: bool = False):
        obs = self._to_tensors(obs)[0]
        logits = self.policy_net(obs)
        if deterministic:
            return logits.argmax(dim=-1).cpu().numpy()
        return Categorical(logits=logits).sample().cpu().numpy()

    def _update_critic(self, *args):
        critic_loss = self._update_critic_compiled(*args)
        critic_loss = self.generic_update(critic_loss, self.critic_optim)
        return critic_loss.item()

    @torch.compile
    def _update_critic_compiled(self, obs, acts, rews, next_obs, dones, flags, next_flags):
        with torch.autocast(device_type="cuda", enabled=self.scaler is not None, dtype=self._scaler_dtype):
            q1_logits, q2_logits = self.critic_net1(obs), self.critic_net2(obs)
            q1_pred_logits, q2_pred_logits = q1_logits.gather(1, acts.long()), q2_logits.gather(1, acts.long())

            with torch.no_grad():
                next_multistep_discount = torch.where(next_flags == 1, 3, 1)
                current_multistep_discount = torch.where(flags == 1, 2, 0)

                next_policy_logits = self.policy_net(next_obs)
                next_policy_probs = torch.softmax(next_policy_logits, dim=-1)

                next_q1_logits, next_q2_logits = self.target_critic_net1(next_obs), self.target_critic_net2(next_obs)
                next_q_logits = torch.min(next_q1_logits, next_q2_logits)

                next_v_logits = (next_policy_probs * (next_q_logits - self._entropy_alpha * torch.log_softmax(next_policy_logits, dim=-1))).sum(dim=-1, keepdim=True)

                r = self._gamma ** current_multistep_discount * rews.float()
                q_target = r + self._gamma ** next_multistep_discount * (1 - dones.float()) * torch.sigmoid(next_v_logits)

            q1_pred = torch.sigmoid(q1_pred_logits)
            q2_pred = torch.sigmoid(q2_pred_logits)
            bellman_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

            # CQL loss
            logsumexp_q1 = torch.logsumexp(q1_logits, dim=1, keepdim=True)
            logsumexp_q2 = torch.logsumexp(q2_logits, dim=1, keepdim=True)
            cql_loss1 = (logsumexp_q1 - q1_pred_logits).mean()
            cql_loss2 = (logsumexp_q2 - q2_pred_logits).mean()
            cql_loss = cql_loss1 + cql_loss2

            total_loss = bellman_loss + self._cql_alpha * cql_loss

        return total_loss

    def _update_actor(self, *args):
        policy_loss = self._update_actor_compiled(*args)
        policy_loss = self.generic_update(policy_loss, self.policy_optim)
        return policy_loss.item()

    @torch.compile
    def _update_actor_compiled(self, obs, acts=None):
        with torch.autocast(device_type="cuda", enabled=self.scaler is not None, dtype=self._scaler_dtype):
            logits = self.policy_net(obs)
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            with torch.no_grad():
                q1_logits, q2_logits = self.critic_net1(obs), self.critic_net2(obs)
                q_logits = torch.min(q1_logits, q2_logits)
            policy_loss = (probs * (self._entropy_alpha * log_probs - q_logits)).sum(dim=1).mean()
        return policy_loss

    def sync_target_networks(self, tau=None):
        tau = tau or self._tau_target
        for target_param, param in zip(self.target_critic_net1.parameters(), self.critic_net1.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)
        for target_param, param in zip(self.target_critic_net2.parameters(), self.critic_net2.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)
