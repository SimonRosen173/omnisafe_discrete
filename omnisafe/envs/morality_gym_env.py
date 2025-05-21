# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Morality Gym environment for OmniSafe."""

from __future__ import annotations

from typing import Any, ClassVar
import gymnasium
import torch
import numpy as np

from omnisafe.envs.core import CMDP, env_register

# Attempt to import the wrapper from morality_gym
try:
    from morality_gym.wrappers.omnisafe_wrapper import OmniSafeMoralityGymWrapper
except ImportError as e:
    print(f"Could not import OmniSafeMoralityGymWrapper: {e}. \
        Please ensure 'morality-gym-tabular' is installed or in PYTHONPATH.")
    raise

# Attempt to import make_experiment from morality_gym experiments
try:
    from experiments.baselines.common.setup import make_experiment
except ImportError as e:
    print(f"Could not import make_experiment: {e}. \
        Please ensure 'morality-gym-tabular/experiments' is accessible in PYTHONPATH.")
    raise


@env_register
class MoralityGymOmniSafeEnv(CMDP):
    """Custom Morality Gym environment for OmniSafe.

    This class uses `make_experiment` from `morality-gym-tabular` to configure
    a specific environment variant and then wraps it with `OmniSafeMoralityGymWrapper`.
    The `env_id` for OmniSafe registration should be in the format:
    "experiment_name::morality_tree_id::repeat_idx"
    (e.g., "Switch3-v0::Trolley-Common-Utilitarian-OrderedUtilityHarm-v0::0")
    """

    # Define _support_envs with examples of valid composite IDs based on
    # your actual experiment JSON configurations.
    # This list is used by OmniSafe's initial environment check.
    _support_envs: ClassVar[list[str]] = [
        "PushOrSwitch-v1::Trolley-Common-DualProcess-Complex-v0::0",
        "PushOrSwitch-v1::Trolley-Common-Utilitarian-OrderedUtilityHarm-v0::0",
        "PushOrSwitch-v1::Trolley-Common-Utilitarian-OrderedOutcomeHarm1-v0::0",
        "SwitchEasy-v1::Trolley-Common-Utilitarian-OrderedUtilityHarm-v0::0",
        "SwitchEasy-v1::Trolley-Common-Utilitarian-OrderedOutcomeHarm1-v0::0",
        "SwitchStandard-v1::Trolley-Common-Utilitarian-OrderedUtilityHarm-v0::0",
        "SwitchStandard-v1::Trolley-Common-Utilitarian-OrderedOutcomeHarm1-v0::0",
        "PushStandard-v1::Trolley-Common-DualProcess-Complex-v0::0",
        "PushStandard-v1::Trolley-Common-Utilitarian-OrderedUtilityHarm-v0::0",
        "PushStandard-v1::Trolley-Common-Utilitarian-OrderedOutcomeHarm1-v0::0",
        "PushSelfSacrifice-v1::Trolley-Common-DualProcess-ComplexAgentHarm-v0::0",
        "PushSelfSacrifice-v1::Trolley-Common-Utilitarian-OrderedUtilityHarm-v0::0",
        "PushSelfSacrifice-v1::Trolley-Common-Utilitarian-OrderedOutcomeHarm2-v0::0",
        "SwitchSelfSacrifice1-v1::Trolley-Common-Utilitarian-OrderedUtilityHarm-v0::0",
        "SwitchSelfSacrifice1-v1::Trolley-Common-Utilitarian-OrderedOutcomeHarm2-v0::0",
        "SwitchSelfSacrifice2-v1::Trolley-Common-Utilitarian-OrderedUtilityHarm-v0::0",
        "SwitchSelfSacrifice2-v1::Trolley-Common-Utilitarian-OrderedOutcomeHarm2-v0::0",
        "PushOrSwitchSelfSacrifice-v1::Trolley-Common-DualProcess-ComplexAgentHarm-v0::0",
        "PushOrSwitchSelfSacrifice-v1::Trolley-Common-Utilitarian-OrderedUtilityHarm-v0::0",
        "PushOrSwitchSelfSacrifice-v1::Trolley-Common-Utilitarian-OrderedOutcomeHarm2-v0::0",
        "Push2Character-v1::Trolley-Common-DualProcess-Complex-v0::0",
        "Push2Character-v1::Trolley-Common-Utilitarian-OrderedUtilityHarm-v0::0",
        "Push2Character-v1::Trolley-Common-Utilitarian-OrderedOutcomeHarm1-v0::0",
        "Switch2Trolley-v1::Trolley-Common-Utilitarian-OrderedUtilityHarm-v0::0",
        "Switch2Trolley-v1::Trolley-Common-Utilitarian-OrderedOutcomeHarm1-v0::0",
        "Switch3-v1::Trolley-Common-Utilitarian-OrderedUtilityHarm-v0::0",
        "Switch3-v1::Trolley-Common-Utilitarian-OrderedOutcomeHarm1-v0::0",
        "Switch7-v1::Trolley-Common-Utilitarian-OrderedUtilityHarm-v0::0",
        "Switch7-v1::Trolley-Common-Utilitarian-OrderedOutcomeHarm1-v0::0",
        "Switch2Character-v1::Trolley-Common-Utilitarian-OrderedUtilityHarm-v0::0",
        "Switch2Character-v1::Trolley-Common-Utilitarian-OrderedOutcomeHarm1-v0::0",
        "Switch2Trolley2Lever-v1::Trolley-Common-Utilitarian-OrderedUtilityHarm-v0::0",
        "Switch2Trolley2Lever-v1::Trolley-Common-Utilitarian-OrderedOutcomeHarm1-v0::0",
        "Switch2TrolleyDistractor-v1::Trolley-Common-Utilitarian-OrderedUtilityHarm-v0::0",
        "Switch2TrolleyDistractor-v1::Trolley-Common-Utilitarian-OrderedOutcomeHarm1-v0::0",
        "Switch2TrolleySelfSacrifice-v1::Trolley-Common-Utilitarian-OrderedUtilityHarm-v0::0",
        "Switch2TrolleySelfSacrifice-v1::Trolley-Common-Utilitarian-OrderedOutcomeHarm1-v0::0",
        "Switch3Trolley3Lever-v1::Trolley-Common-Utilitarian-OrderedUtilityHarm-v0::0",
        "Switch3Trolley3Lever-v1::Trolley-Common-Utilitarian-OrderedOutcomeHarm1-v0::0",
        "Switch3Trolley-v1::Trolley-Common-Utilitarian-OrderedUtilityHarm-v0::0",
        # Add other specific variants for other experiments you intend to run with OmniSafe
        # For example, if you have a Trolley-SwitchStandard-0-v0.json:
        # "Trolley-SwitchStandard-0-v0::SomeTreeForTrolleySwitch::0",
        # For example, if you have a Trolley-PushStandard-v0.json:
        # "Trolley-PushStandard-v0::SomeTreeForTrolleyPush::0",
    ]

    need_action_scale_wrapper = False
    need_obs_normalize_wrapper = False
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = False

    def __init__(
        self,
        env_id: str, # Expected format: "experiment_name::morality_tree_id::repeat_idx"
        num_envs: int = 1,
        device: str = 'cpu',
        seed: int = None,  # Add explicit seed parameter
        **kwargs: Any, # For additional OmniSafe configs like cost_limit
    ) -> None:
        self._num_envs = num_envs
        self._device = device
        self._env_id_passed_to_init = env_id

        try:
            exp_name, tree_id, repeat_str = env_id.split('::')
            repeat_idx = int(repeat_str)
        except ValueError as e:
            raise ValueError(
                f"MoralityGymOmniSafeEnv env_id '{env_id}' is not in the expected format \
                'experiment_name::morality_tree_id::repeat_idx'. Error: {e}"
            )

        all_variants_make_kwargs, _, _ , _ = make_experiment(exp_name)
        
        #print(f"all_variants_make_kwargs: {all_variants_make_kwargs}")
        #exit()
        variant_key = (tree_id, repeat_idx)
        if variant_key not in all_variants_make_kwargs:
            raise ValueError(
                f"Variant ('{tree_id}', {repeat_idx}) not found for experiment '{exp_name}'. \
                Available variants: {list(all_variants_make_kwargs.keys())}"
            )
        
        selected_curr_kwargs = all_variants_make_kwargs[variant_key]

        actual_mg_env_id = selected_curr_kwargs.get('env_id')
        actual_mg_tree_id = selected_curr_kwargs.get('morality_tree_id')
        actual_mg_env_kwargs = selected_curr_kwargs.get('env_kwargs', {})

        if actual_mg_env_id is None or actual_mg_tree_id is None:
            raise ValueError(
                f"The 'curr_kwargs' from make_experiment for '{exp_name}/{variant_key}' \
                must contain 'env_id' and 'morality_tree_id'. Got: {selected_curr_kwargs}"
            )

        # Use the explicit seed parameter if provided, otherwise check kwargs
        if seed is not None:
            # Update the scenario_overrides with the provided seed
            if 'scenario_overrides' not in actual_mg_env_kwargs:
                actual_mg_env_kwargs['scenario_overrides'] = {}
            actual_mg_env_kwargs['scenario_overrides']['seed'] = seed
            print(f"Using explicitly passed seed: {seed}")

        
        # Create the environment wrapper
        self._env: gymnasium.Env = OmniSafeMoralityGymWrapper(
            env_id=actual_mg_env_id,
            morality_tree_id=actual_mg_tree_id,
            env_kwargs=actual_mg_env_kwargs,
        )
        
        self._action_space = self._env.action_space
        self._observation_space = self._env.observation_space
        self.max_episode_steps = getattr(self._env.env, '_max_episode_steps', 1000)

        #default_cost_limit = 0.0
        #cost_limit_from_make_exp = selected_curr_kwargs.get('cost_limit')
        #if 'cost_limit' in kwargs:
        #    self._cost_limit = float(kwargs['cost_limit'])
        #elif cost_limit_from_make_exp is not None:
        #    self._cost_limit = float(cost_limit_from_make_exp)
        #else:
        #    self._cost_limit = default_cost_limit

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        np_action = action.detach().cpu().numpy()
        obs, reward, terminated, truncated, info = self._env.step(np_action)

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device=self._device)
        cost_tensor = torch.as_tensor(info.get('cost', 0.0), dtype=torch.float32, device=self._device)
        terminated_tensor = torch.as_tensor(terminated, dtype=torch.bool, device=self._device)
        truncated_tensor = torch.as_tensor(truncated, dtype=torch.bool, device=self._device)
        
        if terminated or truncated:
            # Store the tensor version of the observation in info dict
            info['final_observation'] = obs_tensor

        return obs_tensor, reward_tensor, cost_tensor, terminated_tensor, truncated_tensor, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        obs, info = self._env.reset(seed=seed, options=options)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        return obs_tensor, info

    def set_seed(self, seed: int) -> None:
        self.reset(seed=seed)

    def render(self) -> Any:
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    @property
    def کامd_name(self) -> str: 
        return self._env_id_passed_to_init

    def sample_action(self) -> torch.Tensor:
        """Sample a random action from the environment's action space.

        Returns:
            torch.Tensor: A random action, converted to a PyTorch tensor.
        """
        # The underlying OmniSafeMoralityGymWrapper inherits the action_space from the base MoralityGym env.
        # We expect self._env.action_space.sample() to return a numpy compatible type.
        action_sample = self._env.action_space.sample()
        # Convert to PyTorch tensor, ensure dtype is appropriate for discrete actions (usually int64)
        # or float32 for continuous. Assuming discrete for now based on typical gym envs.
        # Adjust dtype if your environment has continuous actions.
        if isinstance(self._env.action_space, gymnasium.spaces.Discrete):
            return torch.as_tensor(action_sample, dtype=torch.int64, device=self._device)
        elif isinstance(self._env.action_space, gymnasium.spaces.Box):
             return torch.as_tensor(action_sample, dtype=torch.float32, device=self._device)
        else:
            # Fallback for other space types, might need adjustment
            return torch.as_tensor(action_sample, device=self._device)

    # In __init__, add: self._env_id_passed_to_init = env_id (the registered one)
    # This is a bit of a placeholder; OmniSafe's core env might have a better way.

# Need to add torch and numpy imports if not already present at the top level of the module
# import torch
# import numpy as np
# Ensure these are present if the class uses them directly, which it does.
# The linting might complain if they are only in the OmniSafeMoralityGymWrapper.
# For robustness, let's add them here too.
# import torch
# import numpy as np 