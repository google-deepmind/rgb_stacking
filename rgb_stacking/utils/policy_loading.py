# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility for loading RGB stacking policies saved as TF SavedModel."""

from typing import NamedTuple
import dm_env
import numpy as np
# This reverb dependency is needed since otherwise loading a SavedModel throws
# an error when the ReverbDataset op is not found.
import reverb  # pylint: disable=unused-import
import tensorflow as tf
import tree
import typing_extensions

from rgb_stacking.utils import permissive_model


class _MPOState(NamedTuple):
  counter: tf.Tensor
  actor: tree.Structure[tf.Tensor]
  critic: tree.Structure[tf.Tensor]


@typing_extensions.runtime
class Policy(typing_extensions.Protocol):

  def step(self, timestep: dm_env.TimeStep, state: _MPOState):
    pass

  def initial_state(self) -> _MPOState:
    pass


def policy_from_path(saved_model_path: str) -> Policy:
  """Loads policy from stored TF SavedModel."""
  policy = tf.saved_model.load(saved_model_path)
  # Relax strict requirement with respect to its expected inputs, e.g. in
  # regards to unused arguments.
  policy = permissive_model.PermissiveModel(policy)

  # The loaded policy's step function expects batched data. Wrap it so that it
  # expects unbatched data.
  policy_step_batch_fn = policy.step

  def _expand_batch_dim(x):
    return np.expand_dims(x, axis=0)

  def _squeeze_batch_dim(x):
    return np.squeeze(x, axis=0)

  def policy_step_fn(timestep: dm_env.TimeStep, state: _MPOState):
    timestep_batch = dm_env.TimeStep(
        None, None, None,
        tree.map_structure(_expand_batch_dim, timestep.observation))
    state_batch = tree.map_structure(_expand_batch_dim, state)
    output_batch = policy_step_batch_fn(timestep_batch, state_batch)
    output = tree.map_structure(_squeeze_batch_dim, output_batch)
    return output

  policy.step = policy_step_fn
  return policy


class StatefulPolicyCallable:
  """Object-oriented policy for directly using in dm_control viewer."""

  def __init__(self, policy: Policy):
    self._policy = policy
    self._state = self._policy.initial_state()

  def __call__(self, timestep: dm_env.TimeStep):
    if timestep.step_type == dm_env.StepType.FIRST:
      self._state = self._policy.initial_state()
    (action, _), self._state = self._policy.step(timestep, self._state)
    return action
