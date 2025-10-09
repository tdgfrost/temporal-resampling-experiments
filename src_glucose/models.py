from typing import Tuple, Dict, Optional
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import torch.nn as nn
import torch
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.distributions import Categorical
from tqdm import tqdm
from collections import deque
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution


class CallablePPO(PPO):
    """
    A version of PPO that can be called like a function to get actions.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, obs):
        action, _ = self.predict(obs, deterministic=True)
        return action


class CallableRecurrentPPO(RecurrentPPO):
    """
    A version of PPO that can be called like a function to get actions.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, obs, *args, **kwargs):
        action, lstm_states = self.predict(obs, deterministic=True, *args, **kwargs)
        return action, lstm_states


class MiniGridEncoder(nn.Module):
    def __init__(self, observation_shape, feature_size: int = 128, dropout_p: float = 0.0):
        super().__init__()

        self.fc = nn.Sequential(
            # Scale inputs
            nn.Linear(observation_shape[0], feature_size),
            nn.LayerNorm(feature_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )

        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for Conv2d/Linear with Kaiming normal,
        biases with zeros, Norm layers with weight=1, bias=0.
        The final Linear layer in the decoder is zero-initialized.
        """
        def _init_fn(m: nn.Module):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # apply to all submodules
        self.apply(_init_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shrink_x = self.shrink_obs(x)
        return self.fc(shrink_x)



class PPOMiniGridEncoder(MiniGridEncoder):
    def __init__(self, observation_shape: Tuple[int,], feature_size: int = 128,
                 *args, **kwargs) -> None:
        # Ignore all zeros at the start
        self.shrink_obs = lambda x: x
        super().__init__(observation_shape, feature_size)


class OfflineMiniGridEncoder(MiniGridEncoder):
    def __init__(self, observation_shape: Tuple[int,], feature_size: int = 128,
                 input_scaling: bool = True, *args, **kwargs) -> None:

        # additionally ignore flag channel at the start
        self.shrink_obs = lambda x: x[:, 1:]
        new_obs_shape = (observation_shape[0] - 1,)

        super().__init__(new_obs_shape, feature_size)


class PPOMiniGridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512) -> None: #, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        self.net = PPOMiniGridEncoder(observation_space.shape, feature_size=features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class CustomRecurrentPolicy(RecurrentActorCriticPolicy):
    """
    A custom recurrent policy that applies a transformation to the action logits.
    Transformation: (tanh(logits) + 1) * 15
    This maps the output of the action network to the range [0, 30].
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ✅ Initialize the final layer (action_net) for small initial actions
        # We set the bias to a large negative number.
        # This makes the initial logits negative, pushing tanh(logits) towards -1.
        # The result is an initial action close to `(-1 + 1) * 15 = 0`.
        # The weight initialization helps stabilize learning at the start.
        # if hasattr(self.action_net, "bias"):
            # A value like -5 or -10 is a good starting point
            # nn.init.constant_(self.action_net.bias, -1)

        # It's also good practice to initialize the weights to a small scale
        # nn.init.orthogonal_(self.action_net.weight, 0.01)

    def _get_action_dist_from_latent(
        self, latent_pi: torch.Tensor, latent_sde: Optional[torch.Tensor] = None
    ) -> DiagGaussianDistribution:
        """
        Overrides the base method to apply the custom transformation.

        :param latent_pi: Features from the policy network.
        :param latent_sde: Features for state-dependent exploration (not used here).
        :return: A Gaussian distribution with the transformed mean.
        """
        # Get the raw network output (logits)
        mean_actions = self.action_net(latent_pi)

        # Apply your custom transformation
        transformed_mean_actions = torch.sigmoid(mean_actions) * 30

        # This example assumes a continuous action space (DiagGaussianDistribution).
        # If you have a different action distribution, you'll need to adapt this part.
        if not isinstance(self.action_dist, DiagGaussianDistribution):
            raise ValueError(
                "This custom policy is designed for DiagGaussianDistribution."
            )

        # Return the action distribution using the transformed mean
        return self.action_dist.proba_distribution(
            transformed_mean_actions, self.log_std
        )


class ScaleAction(nn.Module):
    def __init__(self, low: float, high: float):
        super().__init__()
        self.register_buffer('low', torch.tensor(low, dtype=torch.float32))
        self.register_buffer('high', torch.tensor(high, dtype=torch.float32))
        self.device = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale from [0, 1] to [low, high]
        if self.device is None or self.device != x.device:
            self.device = x.device
            self.register_buffer('low', self.low.to(self.device))
            self.register_buffer('high', self.high.to(self.device))
        return x * (self.high - self.low) + self.low


class CustomNet(nn.Module):
    def __init__(self, observation_shape: Tuple[int, int, int], output_size: int, feature_size: int = 128,
                 device: str = 'cpu', action_encoder = None, feature_extractor=None, dropout_p: float = 0.0,
                 has_sigmoid: bool = False, *args, **kwargs) -> None:
        super().__init__()
        self._device = device
        if feature_extractor is None:
            self.encoder = OfflineMiniGridEncoder(observation_shape=observation_shape,
                                                  feature_size=feature_size,
                                                  dropout_p=dropout_p).to(device)
        else:
            self.encoder = feature_extractor

        self.action_encoder = action_encoder
        self._has_actions = action_encoder is not None
        self._has_sigmoid = has_sigmoid

        decoder_input_size = feature_size * (1 + int(self._has_actions))

        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_size, decoder_input_size // 2),
            nn.LayerNorm(decoder_input_size // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            nn.Linear(decoder_input_size // 2, decoder_input_size // 2),
            nn.LayerNorm(decoder_input_size // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            nn.Linear(decoder_input_size // 2, output_size)
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

    def forward(self, x: torch.Tensor, actions: torch.Tensor = None, flags: torch.Tensor = None) -> torch.Tensor:
        x, actions, flags = self._ndarray_to_tensor(x, actions, flags)
        x = x.to(dtype=torch.float32)

        # (N, C, H, W) -> (N, feature_size)
        hidden = self.encoder(x)
        if self._has_actions:
            hidden = torch.concatenate([hidden,
                                        self.action_encoder(actions.view(-1, 1).to(dtype=torch.float32))], dim=-1)

        # (N, feature_size) -> (N, output_size)
        output = self.decoder(hidden)

        if flags is not None:
            output = output.view(x.size(0), 2, -1)
            output = torch.take_along_dim(output, flags.long().unsqueeze(-1), 1).squeeze(1)

        if self._has_sigmoid:
            output = torch.sigmoid(output)

        return output

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


class RecurrentNet(nn.Module):
    def __init__(self, observation_shape: Tuple[int, int, int], output_size: int, feature_size: int = 128,
                 recurrent_hidden_size: int = 128, device: str = 'cpu', action_encoder=None,
                 feature_extractor=None, dropout_p: float = 0.0, has_sigmoid: bool = False, *args, **kwargs) -> None:
        super().__init__()
        self._device = device
        self.recurrent_hidden_size = recurrent_hidden_size

        if feature_extractor is None:
            # This encoder now processes individual frames
            self.encoder = OfflineMiniGridEncoder(observation_shape=observation_shape, feature_size=feature_size,
                                                  dropout_p=dropout_p).to(device)
        else:
            self.encoder = feature_extractor

        self.action_encoder = action_encoder
        self._has_actions = action_encoder is not None
        self._has_sigmoid = has_sigmoid

        # Input to LSTM is the feature vector (plus action embedding if present)
        lstm_input_size = feature_size * (1 + int(self._has_actions))

        # --- Recurrent Layer ---
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=recurrent_hidden_size,
            num_layers=1,
            batch_first=True  # Crucial for easier tensor manipulation
        ).to(device)

        # Decoder now takes the output of the LSTM
        decoder_input_size = recurrent_hidden_size
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_size, decoder_input_size // 2),
            nn.LayerNorm(decoder_input_size // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(decoder_input_size // 2, output_size)
        ).to(device)

        # Note: Weight initialization is omitted for brevity but should be the same as in your original code.
        # self.init_weights()

    def forward(self, x: torch.Tensor, actions: torch.Tensor = None,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x shape: (N, T, C, H, W), where T is sequence length
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Reshape to process each frame through the encoder
        # (N, T, C, H, W) -> (N * T, C, H, W)
        x_reshaped = x.reshape(-1, *x.shape[2:])

        # (N * T, C, H, W) -> (N * T, feature_size)
        hidden = self.encoder(x_reshaped.to(dtype=torch.float32))

        if self._has_actions:
            # actions shape: (N, T, 1) -> (N * T, 1)
            actions_reshaped = actions.reshape(-1, 1).to(dtype=torch.float32)
            action_embedding = self.action_encoder(actions_reshaped)
            hidden = torch.cat([hidden, action_embedding], dim=-1)

        # Reshape back to sequence format for LSTM
        # (N * T, lstm_input_size) -> (N, T, lstm_input_size)
        lstm_input = hidden.view(batch_size, seq_len, -1)

        # Process sequence through LSTM
        # lstm_out shape: (N, T, recurrent_hidden_size)
        # next_hidden_state is a tuple (h_n, c_n)
        lstm_out, next_hidden_state = self.lstm(lstm_input, hidden_state)

        # Reshape for the decoder
        # (N, T, recurrent_hidden_size) -> (N * T, recurrent_hidden_size)
        decoder_input = lstm_out.reshape(-1, self.recurrent_hidden_size)

        # (N * T, recurrent_hidden_size) -> (N * T, output_size)
        output = self.decoder(decoder_input)

        # Reshape final output back to sequence format
        # (N * T, output_size) -> (N, T, output_size)
        output = output.view(batch_size, seq_len, -1)

        if self._has_sigmoid:
            output = torch.sigmoid(output)

        return output, next_hidden_state


class CustomIQL(nn.Module):
    def __init__(
            self,
            observation_shape: Tuple[int, int, int],
            action_space: spaces.Box,
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
            device: str = 'cpu'
    ):
        super().__init__()
        self._feature_size = feature_size
        self._batch_size = batch_size
        self._expectile = expectile
        self._gamma = torch.tensor(gamma, device=device)
        self._batch_diff = None
        self._input_length = input_length
        self._device = device
        self._cloning_only = expectile == 0.5
        self._tau_target = tau_target
        self._has_dropout = dropout_p > 0.0
        self._beta = beta
        self._action_low = action_space.low[0]
        self._action_high = action_space.high[0]
        self._critic_lr = critic_lr
        self._value_lr = value_lr
        self._actor_lr = actor_lr

        net_kwargs = dict(observation_shape=observation_shape, feature_size=feature_size, dropout_p=dropout_p, device=device)

        self.feature_extractor = OfflineMiniGridEncoder(**net_kwargs).to(device)
        self.action_encoder = nn.Sequential(
            ScaleAction(low=action_space.low, high=action_space.high),
            nn.Linear(1, feature_size),
            nn.LayerNorm(feature_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        ).to(device)

        self.critic_net1 = CustomNet(output_size=2, action_encoder=self.action_encoder, feature_extractor=self.feature_extractor, **net_kwargs)
        self.critic_net2 = CustomNet(output_size=2, action_encoder=self.action_encoder, feature_extractor=self.feature_extractor, **net_kwargs)

        self.value_net = CustomNet(output_size=2, feature_extractor=self.feature_extractor, **net_kwargs)
        self.target_value_net = CustomNet(output_size=2, feature_extractor=self.feature_extractor, **net_kwargs)

        self.policy_net = CustomNet(output_size=2, feature_extractor=self.feature_extractor, has_sigmoid=True, **net_kwargs)

        # Give both critic nets to the critic optimizer
        self.critic_optim = torch.optim.AdamW(list(self.critic_net1.parameters()) +
                                              list(self.critic_net2.parameters()), lr=critic_lr)
        self.value_optim = torch.optim.AdamW(self.value_net.parameters(), lr=value_lr)
        self.policy_optim = torch.optim.AdamW(self.policy_net.parameters(), lr=actor_lr)

        # clone the value net to a target network
        for target_param, param in zip(self.target_value_net.decoder.parameters(), self.value_net.decoder.parameters()):
            target_param.data.copy_(param.data)

    def fit(
        self,
        dataset,
        n_epochs_train: int = 1,
        n_epochs_eval: int = 1,
        evaluators=None,
        show_progress: bool = True,
        experiment_name: str = None,
        decoy_interval: int = 0,
        dataset_kwargs: Optional[Dict] = None
    ):
        # Initialise our dataset and loss dictionary
        dataset_kwargs = dict() if dataset_kwargs is None else dataset_kwargs
        dataset.set_to_tensors(self._device)
        dataset.set_generate_params(**dataset_kwargs)
        self.decoy_interval = decoy_interval

        loss_dict = self._reset_loss_dict()

        epoch_eval_interval = n_epochs_train // n_epochs_eval

        # Start training
        with tqdm(total=n_epochs_train * len(dataset), desc="Progress", mininterval=2.0, disable=not show_progress) as pbar:
            for epoch in range(1, n_epochs_train + 1):
                epoch_str = f"{epoch}/{n_epochs_train}"

                for batch in dataset:
                    obs, acts, rews, next_obs, dones, visible, next_visible = batch

                    # Update the networks
                    if not self._cloning_only:
                        loss_dict['critic_loss'].append(self._update_critic(obs, acts, rews, next_obs, dones, visible, next_visible))
                        loss_dict['value_loss'].append(self._update_value(obs, acts, visible))
                    loss_dict['policy_loss'].append(self._update_actor(obs, acts, visible))

                    # Soft update of target value network
                    for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
                        target_param.data.copy_((1-self._tau_target) * target_param.data + self._tau_target * param.data)

                    pbar.update(dataset.batch_size)
                    pbar.set_postfix(epoch=epoch_str,
                                     policy_loss=f"{np.mean(loss_dict['policy_loss']):.5f}",
                                     critic_loss=f"{np.mean(loss_dict['critic_loss']):.5f}",
                                     value_loss=f"{np.mean(loss_dict['value_loss']):.5f}",
                                     refresh=False)

                # Logging
                if epoch % epoch_eval_interval == 0 and evaluators is not None:
                    loss_dict, log_dict = self._log_progress(
                        epoch=epoch,
                        loss_dict=loss_dict,
                        experiment_name=experiment_name,
                        evaluators=evaluators
                    )

        return log_dict

    def forward(self, obs, acts, flags=None):
        if flags is None:
            flags = self._extract_flags(obs)
        q1, q2 = self.critic_net1(obs, acts, flags=flags), self.critic_net2(obs, acts, flags=flags)
        v = self.value_net(obs, flags=flags)
        logits = self.policy_net(obs, flags=flags)
        return (q1, q2), v, logits

    def predict(self, obs, flags=None, deterministic: bool = False):
        obs = self._to_tensors(obs)[0]
        if flags is None:
            flags = self._extract_flag(obs)
        action_unit = self.policy_net(obs, flags=flags)
        # Scale up
        action_unit = action_unit * (self._action_high - self._action_low) + self._action_low
        # if deterministic:
            # return logits.argmax(dim=-1).cpu().numpy()
        # return Categorical(logits=logits).sample().cpu().numpy()
        return action_unit.squeeze(-1).cpu().numpy()

    def _update_critic(self, obs, acts, rews, next_obs, dones, flags=None, next_flags=None):
        q1, q2 = self.critic_net1(obs, acts, flags=flags), self.critic_net2(obs, acts, flags=flags)
        with torch.no_grad():
            v_next = self.target_value_net(next_obs, flags=next_flags)
            # next_multistep_discount = torch.where(next_flags == 1, 3, 1)
            # current_multistep_discount = torch.where(flags == 1, 2, 0)
            # r = self._gamma ** current_multistep_discount * rews.float()
            r = rews.float()
            next_multistep_discount = torch.where(flags == 1, 6, 1)
            q_target = r + self._gamma ** next_multistep_discount * (1 - dones.float()) * v_next

        loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        return loss.item()

    def _update_value(self, obs, acts, flags=None):
        v = self.value_net(obs, flags=flags)
        with torch.no_grad():
            q1, q2 = self.critic_net1(obs, acts, flags=flags), self.critic_net2(obs, acts, flags=flags)
            q = torch.min(q1, q2)

        diff = q - v
        self._batch_diff = diff.detach().squeeze()
        weights = torch.absolute(self._expectile - (self._batch_diff < 0).float()).squeeze()
        value_loss = (weights * (diff.squeeze() ** 2)).mean()

        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()
        return value_loss.item()

    def _update_actor(self, obs, acts, flags=None):
        action_unit_preds = self.policy_net(obs, flags=flags)
        predicted_actions = action_unit_preds * (self._action_high - self._action_low) + self._action_low

        weights = 1.0
        if not self._cloning_only:
            weights = self._batch_diff
            if self._has_dropout:
                weights = torch.absolute(self._expectile - (self._batch_diff < 0).float()).squeeze()
            else:
                weights = torch.clip(torch.exp(self._beta * weights), -torch.inf, 100).squeeze()

        # policy_loss = F.cross_entropy(logits, acts.squeeze().long(), reduction='none')
        policy_loss = F.mse_loss(predicted_actions.squeeze(), acts.squeeze(), reduction='none')
        policy_loss = (policy_loss * weights).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        return policy_loss.item()

    def _extract_flag(self, obs):
        obs = self._to_tensors(obs)[0]
        return obs[..., 0].unsqueeze(-1).long()

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
            mean_rew, std_rew = evaluators[key](self)
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


class RecurrentIQL(CustomIQL):  # Inherits from your original class
    def __init__(self, *args, sequence_length: int = 10, recurrent_hidden_size: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        self._sequence_length = sequence_length
        self._recurrent_hidden_size = recurrent_hidden_size

        # --- Replace network instantiations with RecurrentNet ---
        net_kwargs = dict(
            observation_shape=kwargs.get('observation_shape'),
            feature_size=self._feature_size,
            recurrent_hidden_size=recurrent_hidden_size,
            dropout_p=kwargs.get('dropout_p', 0.0),
            device=self._device
        )

        # Re-initialize networks with RecurrentNet
        self.critic_net1 = RecurrentNet(output_size=1, action_encoder=self.action_encoder,
                                        feature_extractor=self.feature_extractor, **net_kwargs)
        self.critic_net2 = RecurrentNet(output_size=1, action_encoder=self.action_encoder,
                                        feature_extractor=self.feature_extractor, **net_kwargs)
        self.value_net = RecurrentNet(output_size=1, feature_extractor=self.feature_extractor, **net_kwargs)
        self.target_value_net = RecurrentNet(output_size=1, feature_extractor=self.feature_extractor, **net_kwargs)
        self.policy_net = RecurrentNet(output_size=1, feature_extractor=self.feature_extractor, has_sigmoid=True,
                                       **net_kwargs)

        # Give both critic nets to the critic optimizer
        self.critic_optim = torch.optim.AdamW(list(self.critic_net1.parameters()) +
                                              list(self.critic_net2.parameters()), lr=self._critic_lr)
        self.value_optim = torch.optim.AdamW(self.value_net.parameters(), lr=self._value_lr)
        self.policy_optim = torch.optim.AdamW(self.policy_net.parameters(), lr=self._actor_lr)

        # clone the value net to a target network
        for target_param, param in zip(self.target_value_net.decoder.parameters(), self.value_net.decoder.parameters()):
            target_param.data.copy_(param.data)

    def get_initial_states(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a zero-initialized hidden state tuple for the LSTM."""
        h_0 = torch.zeros(1, batch_size, self._recurrent_hidden_size, device=self._device)
        c_0 = torch.zeros(1, batch_size, self._recurrent_hidden_size, device=self._device)
        return (h_0, c_0)

    def predict(self, obs: np.ndarray, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                deterministic: bool = False) -> Tuple[np.ndarray, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predicts an action for a single observation and manages the recurrent state.
        Returns the action and the next hidden state.
        """
        obs_tensor = self._to_tensors(obs)[0]
        # Add batch and sequence dimensions: (L,W) -> (N, L,W)
        obs_tensor = obs_tensor.unsqueeze(0)

        # Pass the current hidden_state to the policy network
        action_unit, next_hidden_state = self.policy_net(obs_tensor, hidden_state=hidden_state)

        # Remove sequence dimension for scaling
        action_unit = action_unit.squeeze(1)

        # Scale up
        action = action_unit * (self._action_high - self._action_low) + self._action_low

        return action.squeeze(-1).cpu().numpy(), next_hidden_state

    # --- Update methods now handle sequences ---
    # The core logic remains the same, but we unpack the network output

    def _update_critic(self, obs, acts, rews, next_obs, dones, visible=None, next_visible=None):
        # We only need the network output, not the final hidden state for training
        q1, _ = self.critic_net1(obs, acts)
        q2, _ = self.critic_net2(obs, acts)

        with torch.no_grad():
            v_next, _ = self.target_value_net(next_obs)
            r = rews.float()

        if self.decoy_interval == 0:
            q1, q2, q_target = self._filter_to_correct_visibles(q1, q2, r, v_next, dones, visible, next_visible)
        else:
            q_target = r + self._gamma * (1 - dones.float()) * v_next

        loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        return loss.item()


    # The _update_value and _update_actor methods are updated similarly
    def _update_value(self, obs, acts, visible=None):
        v, _ = self.value_net(obs)
        with torch.no_grad():
            q1, _ = self.critic_net1(obs, acts)
            q2, _ = self.critic_net2(obs, acts)
            q = torch.min(q1, q2)
        # ... rest of the function is the same ...
        diff = q - v
        self._batch_diff = diff.detach()  # Shape is now (N, T, 1)
        weights = torch.absolute(self._expectile - (self._batch_diff < 0).float())
        value_loss = (weights * (diff ** 2))
        # Apply mask
        if self.decoy_interval == 0:
            value_loss = value_loss[visible.squeeze(-1)]
        value_loss = value_loss.mean()

        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()
        return value_loss.item()

    def _update_actor(self, obs, acts, visible=None):
        action_unit_preds, _ = self.policy_net(obs)
        predicted_actions = action_unit_preds * (self._action_high - self._action_low) + self._action_low
        # ... rest of the function is similar ...
        weights = 1.0
        if not self._cloning_only:
            weights = self._batch_diff
            if self._has_dropout:
                weights = torch.absolute(self._expectile - (self._batch_diff < 0).float()).squeeze()
            else:
                weights = torch.clip(torch.exp(self._beta * weights), -100, 100).squeeze()

        policy_loss = F.mse_loss(predicted_actions, acts.view(predicted_actions.shape), reduction='none').squeeze()
        policy_loss = (policy_loss * weights)
        # Apply mask
        if self.decoy_interval == 0:
            policy_loss = policy_loss[visible.squeeze(-1)]
        policy_loss = policy_loss.mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        return policy_loss.item()

    def _filter_to_correct_visibles(self, q1, q2, r, v_next, dones, visible, next_visible):
        # shapes: q1,q2 (N,T,1 or N,T,*), r,v_next,dones,visible,next_visible (N,T,1)
        N, T = visible.shape[:2]
        dev = q1.device
        dtype = r.dtype

        vis = visible.squeeze(-1).bool()  # (N,T)
        nvis = next_visible.squeeze(-1).bool()  # (N,T)
        dn = dones.squeeze(-1).bool()  # (N,T)

        t_idx = torch.arange(T, device=dev).unsqueeze(0).expand(N, T)
        none = torch.full((N, T), T, device=dev)

        def next_true_idx(mask):  # earliest j >= t if exists else T
            cand = torch.where(mask, t_idx, none)
            rev = torch.flip(cand, dims=[1])
            suf = torch.cummin(rev, dim=1).values
            return torch.flip(suf, dims=[1])

        next_done = next_true_idx(dn)
        next_nv = next_true_idx(nvis)

        # prefer next_visible unless dones occurs earlier or at the same time
        target_idx = torch.where(next_done <= next_nv, next_done, next_nv)  # (N,T)
        valid = vis & (target_idx < T)

        # gather helpers
        b_full = torch.arange(N, device=dev).unsqueeze(1).expand(N, T)
        b = b_full[valid]  # (M,)
        i = t_idx[valid]  # (M,)
        j = target_idx[valid]  # (M,)

        # select current Q-values at visible timesteps
        q1_sel = q1[b, i]
        q2_sel = q2[b, i]

        # n-step reward sum over [i..j] with constant gamma
        r_flat = r.squeeze(-1)  # (N,T)
        pow_t = torch.pow(self._gamma, torch.arange(T, device=dev, dtype=dtype))  # (T,)
        rg = r_flat * pow_t  # r_t * γ^t
        S = torch.cumsum(rg, dim=1)  # prefix sum of r_t * γ^t

        Sj = S[b, j]
        Si_1 = torch.where(i > 0, S[b, i - 1], torch.zeros_like(Sj))
        gamma_neg_i = torch.pow(self._gamma, -i.to(dtype))
        R_i_j = gamma_neg_i * (Sj - Si_1)  # Σ_{k=i..j} γ^{k-i} r_k   (M,)

        # bootstrap from s_{j+1} with γ^(j-i+1) and (1 - done_j)
        vnext_j = v_next[b, j].squeeze(-1)  # (M,)
        done_j = dn[b, j].to(dtype)  # (M,)
        expo = torch.pow(self._gamma,
                         (j - i + 1).to(dtype))  # (M,)

        q_target_sel = R_i_j.unsqueeze(-1) + (expo * (1 - done_j) * vnext_j).unsqueeze(-1)

        return q1_sel, q2_sel, q_target_sel
