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

"""Generic reward functions."""

from typing import Callable, Optional, Iterable
from absl import logging
from dm_robotics import agentflow as af
from dm_robotics.agentflow import spec_utils
import numpy as np

# Value returned by the gripper grasp observation when the gripper is in the
# 'grasp' state (fingers closed and exerting force)
_INWARD_GRASP = 2

# Minimal value for the position tolerance of the shaped distance rewards.
MINIMUM_POSITION_TOLERANCE = 1e-9

RewardFunction = Callable[[spec_utils.ObservationValue], float]


class RewardPreprocessor(af.TimestepPreprocessor):
  """Timestep preprocessor wrapper around a reward function."""

  def __init__(self, reward_function: RewardFunction):
    super().__init__()
    self._reward_function = reward_function

  def _process_impl(
      self, timestep: af.PreprocessorTimestep) -> af.PreprocessorTimestep:
    reward = self._reward_function(timestep.observation)
    reward = self._out_spec.reward_spec.dtype.type(reward)
    return timestep.replace(reward=reward)

  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    return input_spec


class GraspReward:
  """Sparse reward for the gripper grasp status."""

  def __init__(self, obs_key: str):
    """Creates new GraspReward function.

    Args:
      obs_key: Key of the grasp observation in the observation spec.
    """
    self._obs_key = obs_key

  def __call__(self, obs: spec_utils.ObservationValue):
    is_grasped = obs[self._obs_key][0] == _INWARD_GRASP
    return float(is_grasped)


class StackPair:
  """Get two objects to be above each other.

  To determine the expected position the top object should be at, the size of
  the objects must be specified. Currently, only objects with 3-axial symmetry
  are supported; for other objects, the distance would impose a constraint on
  the possible orientations.

  Reward will be given if the top object's center is within a certain distance
  of the point above the bottom object, with the distance tolerance being
  separately configurable for the horizontal plane and the vertical axis.

  By default, reward is also only given when the robot is not currently grasping
  anything, though this can be deactivated.
  """

  def __init__(self,
               obj_top: str,
               obj_bottom: str,
               obj_top_size: float,
               obj_bottom_size: float,
               horizontal_tolerance: float,
               vertical_tolerance: float,
               grasp_key: str = "gripper/grasp"):
    """Initialize the module.

    Args:
      obj_top: Key in the observation dict containing the top object's
          position.
      obj_bottom: Key in the observation dict containing the bottom
          object's position.
      obj_top_size: Height of the top object along the vertical axis, in meters.
      obj_bottom_size: Height of the bottom object along the vertical axis, in
          meters.
      horizontal_tolerance: Max distance from the exact stack position on the
          horizontal plane to still generate reward.
      vertical_tolerance: Max distance from the exact stack position along the
          vertical axis to still generate reward.
      grasp_key: Key in the observation dict containing the (buffered) grasp
          status. Can be set to `None` to not check the grasp status to return
          a reward.
    """
    self._top_key = obj_top
    self._bottom_key = obj_bottom
    self._horizontal_tolerance = horizontal_tolerance
    self._vertical_tolerance = vertical_tolerance
    self._grasp_key = grasp_key

    if obj_top_size <= 0. or obj_bottom_size <= 0.:
      raise ValueError("Object sizes cannot be zero.")
    self._expected_dist = (obj_top_size + obj_bottom_size) / 2.

  def __call__(self, obs: spec_utils.ObservationValue):
    top = obs[self._top_key]
    bottom = obs[self._bottom_key]

    horizontal_dist = np.linalg.norm(top[:2] - bottom[:2])
    if horizontal_dist > self._horizontal_tolerance:
      return 0.

    vertical_dist = top[2] - bottom[2]
    if np.abs(vertical_dist - self._expected_dist) > self._vertical_tolerance:
      return 0.

    if self._grasp_key is not None:
      grasp = obs[self._grasp_key]
      if grasp == _INWARD_GRASP:
        return 0.

    return 1.


def tanh_squared(x: np.ndarray, margin: float, loss_at_margin: float = 0.95):
  """Returns a sigmoidal shaping loss based on Hafner & Reidmiller (2011).

  Args:
    x: A numpy array representing the error.
    margin: Margin parameter, a positive `float`.
    loss_at_margin: The loss when `l2_norm(x) == margin`. A `float` between 0
      and 1.

  Returns:
    Shaping loss, a `float` bounded in the half-open interval [0, 1).

  Raises:
    ValueError: If the value of `margin` or `loss_at_margin` is invalid.
  """

  if not margin > 0:
    raise ValueError("`margin` must be positive.")
  if not 0.0 < loss_at_margin < 1.0:
    raise ValueError("`loss_at_margin` must be between 0 and 1.")

  error = np.linalg.norm(x)
  # Compute weight such that at the margin tanh(w * error) = loss_at_margin
  w = np.arctanh(np.sqrt(loss_at_margin)) / margin
  s = np.tanh(w * error)
  return s * s


class DistanceReward:
  """Shaped reward based on the distance B-A between two entities A and B."""

  def __init__(
      self,
      key_a: str,
      key_b: Optional[str],
      position_tolerance: Optional[np.ndarray] = None,
      shaping_tolerance: float = 0.1,
      loss_at_tolerance: float = 0.95,
      max_reward: float = 1.,
      offset: Optional[np.ndarray] = None,
      z_min: Optional[float] = None,
      dim=3
  ):
    """Initialize the module.

    Args:
      key_a: Observation dict key to numpy array containing the position of
          object A.
      key_b: None or observation dict key to numpy array containing the position
          of object B. If None, distance simplifies to d = offset - A.
      position_tolerance: Vector of length `dim`. If
          `distance/position_tolerance < 1`, will return `maximum_reward`
          instead of shaped one. Setting this to `None`, or setting any entry
          to zero or close to zero, will effectively disable tolerance.
      shaping_tolerance: Scalar distance at which the loss is equal to
          `loss_at_tolerance`. Must be a positive float or `None`. If `None`
          reward is sparse and hence 0 is returned if
          `distance > position_tolerance`.
      loss_at_tolerance: The loss when `l2_norm(distance) == shaping_tolerance`.
          A `float` between 0 and 1.
      max_reward: Reward to return when `distance/position_tolerance < 1`.
      offset: Vector of length 3 that is added to the distance, i.e.
          `distance = B - A + offset`.
      z_min: Absolute object height that the object A center has to above be in
          order to generate reward. Used for example in hovering rewards.
      dim: The dimensionality of the space in which the distance is computed
    """
    self._key_a = key_a
    self._key_b = key_b

    self._shaping_tolerance = shaping_tolerance
    self._loss_at_tolerance = loss_at_tolerance
    if max_reward < 1.:
      logging.warning("Maximum reward should not be below tanh maximum.")
    self._max_reward = max_reward
    self._z_min = z_min
    self._dim = dim

    if position_tolerance is None:
      self._position_tolerance = np.full(
          (dim,), fill_value=MINIMUM_POSITION_TOLERANCE)
    else:
      self._position_tolerance = position_tolerance
      self._position_tolerance[self._position_tolerance == 0] = (
          MINIMUM_POSITION_TOLERANCE)

    if offset is None:
      self._offset = np.zeros((dim,))
    else:
      self._offset = offset

  def __call__(self, obs: spec_utils.ObservationValue) -> float:

    # Check that object A is high enough before computing the reward.
    if self._z_min is not None and obs[self._key_a][2] < self._z_min:
      return 0.

    self._current_distance = (self._offset - obs[self._key_a][0:self._dim])
    if self._key_b is not None:
      self._current_distance += obs[self._key_b][0:self._dim]

    weighted = self._current_distance / self._position_tolerance
    if np.linalg.norm(weighted) <= 1.:
      return self._max_reward

    if not self._shaping_tolerance:
      return 0.

    loss = tanh_squared(
        self._current_distance, margin=self._shaping_tolerance,
        loss_at_margin=self._loss_at_tolerance)
    return 1.0 - loss


class LiftShaped:
  """Linear shaped reward for lifting, up to a specified height.

  Once the height is above a specified threshold, reward saturates. Shaping can
  also be deactivated for a sparse reward.

  Requires an observation `<obs_prefix>/<obj_name>/abs_pose` containing the
  pose of the object in question.
  """

  def __init__(
      self,
      obj_key,
      z_threshold,
      z_min,
      max_reward=1.,
      shaping=True
  ):
    """Initialize the module.

    Args:
      obj_key: Key in the observation dict containing the object pose.
      z_threshold: Absolute object height at which the maximum reward will be
          given.
      z_min: Absolute object height that the object center has to above be in
          order to generate shaped reward. Ignored if `shaping` is False.
      max_reward: Reward given when the object is above the `z_threshold`.
      shaping: If true, will give a linear shaped reward when the object height
          is above `z_min`, but below `z_threshold`.
    Raises:
      ValueError: if `z_min` is larger than `z_threshold`.
    """
    self._field = obj_key
    self._z_threshold = z_threshold
    self._shaping = shaping
    self._max_reward = max_reward
    self._z_min = z_min
    if z_min > z_threshold:
      raise ValueError("Lower shaping bound cannot be below upper bound.")

  def __call__(self, obs: spec_utils.ObservationValue) -> float:
    obj_z = obs[self._field][2]
    if obj_z <= self._z_min:
      return 0.0
    if obj_z >= self._z_threshold:
      return self._max_reward
    if self._shaping:
      r = (obj_z - self._z_min) / (self._z_threshold - self._z_min)
      return r
    return 0.0


class Product:
  """Computes the product of a set of rewards."""

  def __init__(self, terms: Iterable[RewardFunction]):
    self._terms = terms

  def __call__(self, obs: spec_utils.ObservationValue) -> float:
    r = 1.
    for term in self._terms:
      r *= term(obs)
    return r


class _WeightedTermReward:
  """Base class for rewards using lists of weighted terms."""

  def __init__(self,
               terms: Iterable[RewardFunction],
               weights: Optional[Iterable[float]] = None):
    """Initialize the reward instance.

    Args:
      terms: List of reward callables to be operated on. Each callable must
          take an observation as input, and return a float.
      weights: Weight that each reward returned by the callables in `terms` will
          be multiplied by. If `None`, will weight all terms with 1.0.
    Raises:
      ValueError: If `weights` has been specified, but its length differs from
          that of `terms`.
    """
    self._terms = list(terms)
    self._weights = weights or [1.] * len(self._terms)
    if len(self._weights) != len(self._terms):
      raise ValueError("Number of terms and weights should be same.")

  def _weighted_terms(
      self, obs: spec_utils.ObservationValue) -> Iterable[float]:
    return [t(obs) * w for t, w in zip(self._terms, self._weights)]


class Max(_WeightedTermReward):
  """Selects the maximum among a number of weighted rewards."""

  def __call__(self, obs: spec_utils.ObservationValue) -> float:
    return max(self._weighted_terms(obs))


class ConditionalAnd:
  """Perform an and operation conditional on term1 exceeding a threshold."""

  def __init__(self,
               term1: RewardFunction,
               term2: RewardFunction,
               threshold: float):
    self._term1 = term1
    self._term2 = term2
    self._thresh = threshold

  def __call__(self, obs: spec_utils.ObservationValue) -> float:
    r1 = self._term1(obs)
    r2 = self._term2(obs)
    if r1 > self._thresh:
      return (0.5 + r2 / 2.) * r1
    else:
      return r1 * 0.5


class Staged:
  """Stages the rewards.

  This works by cycling through the terms backwards and using the last reward
  that gives a response above the provided threshold + the number of
  terms preceding it.

  Rewards must be in [0;1], otherwise they will be clipped.
  """

  def __init__(
      self, terms: Iterable[RewardFunction], threshold: float):
    def make_clipped(term: RewardFunction):
      return lambda obs: np.clip(term(obs), 0., 1.)
    self._terms = [make_clipped(term) for term in terms]
    self._thresh = threshold

  def __call__(self, obs: spec_utils.ObservationValue) -> float:
    last_reward = 0.
    num_stages = float(len(self._terms))
    for i, term in enumerate(reversed(self._terms)):
      last_reward = term(obs)
      if last_reward > self._thresh:
        # Found a reward above the threshold, add number of preceding terms
        # and normalize with the number of terms.
        return (len(self._terms) - (i + 1) + last_reward) / num_stages
    # Return the accumulated rewards.
    return last_reward / num_stages
