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

"""Shaped and sparse reward functions for the RGB stacking task."""

from typing import Callable

from dm_control import mjcf
from dm_robotics.agentflow import spec_utils
from dm_robotics.moma import prop
import numpy as np

from rgb_stacking import physics_utils
from rgb_stacking import reward_functions
from rgb_stacking import task

# Heights above basket surface between which shaping for Lift will be computed.
_LIFT_Z_MAX_ABOVE_BASKET = 0.1
_LIFT_Z_MIN_ABOVE_BASKET = 0.055

# Height for lifting.
_LIFTING_HEIGHT = 0.04

# Cylinder around (bottom object + object height) in which the top object must
# be for it to be considered stacked.
_STACK_HORIZONTAL_TOLERANCE = 0.03
_STACK_VERTICAL_TOLERANCE = 0.01

# Target height above bottom object for hover-above reward.
_HOVER_OFFSET = 0.1

# Distances at which maximum reach reward is given and shaping decays.
_REACH_POSITION_TOLERANCE = np.array((0.02, 0.02, 0.02))
_REACH_SHAPING_TOLERANCE = 0.15

# Distances at which maximum hovering reward is given and shaping decays.
_HOVER_POSITION_TOLERANCE = np.array((0.01, 0.01, 0.01))
_HOVER_SHAPING_TOLERANCE = 0.2

# Observation keys needed to compute the different rewards.
_PINCH_POSE = 'sawyer/pinch/pose'
_FINGER_ANGLE = 'gripper/joints/angle'
_GRIPPER_GRASP = 'gripper/grasp'
_RED_OBJECT_POSE = 'rgb30_red/abs_pose'
_BLUE_OBJECT_POSE = 'rgb30_blue/abs_pose'


# Object closeness threshold for x and y axis.
_X_Y_CLOSE = 0.05
_ONTOP = 0.02


def get_shaped_stacking_reward() -> reward_functions.RewardFunction:
  """Returns a callable reward function for stacking the red block on blue."""
  # # First stage: reach and grasp.
  reach_red = reward_functions.DistanceReward(
      key_a=_PINCH_POSE,
      key_b=_RED_OBJECT_POSE,
      shaping_tolerance=_REACH_SHAPING_TOLERANCE,
      position_tolerance=_REACH_POSITION_TOLERANCE)
  close_fingers = reward_functions.DistanceReward(
      key_a=_FINGER_ANGLE,
      key_b=None,
      position_tolerance=None,
      shaping_tolerance=255.,
      loss_at_tolerance=0.95,
      max_reward=1.,
      offset=np.array((255,)),
      dim=1)
  grasp = reward_functions.Max(
      terms=(
          close_fingers, reward_functions.GraspReward(obs_key=_GRIPPER_GRASP)),
      weights=(0.5, 1.))
  reach_grasp = reward_functions.ConditionalAnd(reach_red, grasp, 0.9)

  # Second stage: grasp and lift.
  lift_reward = _get_reward_lift_red()
  lift_red = reward_functions.Product([grasp, lift_reward])

  # Third stage: hover.
  top = _RED_OBJECT_POSE
  bottom = _BLUE_OBJECT_POSE
  place = reward_functions.DistanceReward(
      key_a=top,
      key_b=bottom,
      offset=np.array((0., 0., _LIFTING_HEIGHT)),
      position_tolerance=_HOVER_POSITION_TOLERANCE,
      shaping_tolerance=_HOVER_SHAPING_TOLERANCE,
      z_min=0.1)

  # Fourth stage: stack.
  stack = _get_reward_stack_red_on_blue()

  # Final stage: stack-and-leave
  stack_leave = reward_functions.Product(
      terms=(_get_reward_stack_red_on_blue(), _get_reward_above_red()))

  return reward_functions.Staged(
      [reach_grasp, lift_red, place, stack, stack_leave], 0.01)


def get_sparse_stacking_reward() ->reward_functions.RewardFunction:
  """Sparse stacking reward for red-on-blue with the gripper moved away."""
  return reward_functions.Product(
      terms=(_get_reward_stack_red_on_blue(), _get_reward_above_red()))


def _get_reward_lift_red() -> reward_functions.RewardFunction:
  """Returns a callable reward function for lifting the red block."""
  lift_reward = reward_functions.LiftShaped(
      obj_key=_RED_OBJECT_POSE,
      z_threshold=task.DEFAULT_BASKET_HEIGHT + _LIFT_Z_MAX_ABOVE_BASKET,
      z_min=task.DEFAULT_BASKET_HEIGHT + _LIFT_Z_MIN_ABOVE_BASKET)
  # Keep the object inside the area of the base plate.
  inside_basket = reward_functions.DistanceReward(
      key_a=_RED_OBJECT_POSE,
      key_b=None,
      position_tolerance=task.WORKSPACE_SIZE / 2.,
      shaping_tolerance=1e-12,  # Practically none.
      loss_at_tolerance=0.95,
      max_reward=1.,
      offset=task.WORKSPACE_CENTER)
  return reward_functions.Product([lift_reward, inside_basket])


def _get_reward_stack_red_on_blue() -> reward_functions.RewardFunction:
  """Returns a callable reward function for stacking the red block on blue."""
  return reward_functions.StackPair(
      obj_top=_RED_OBJECT_POSE,
      obj_bottom=_BLUE_OBJECT_POSE,
      obj_top_size=_LIFTING_HEIGHT,
      obj_bottom_size=_LIFTING_HEIGHT,
      horizontal_tolerance=_STACK_HORIZONTAL_TOLERANCE,
      vertical_tolerance=_STACK_VERTICAL_TOLERANCE)


def _get_reward_above_red() -> reward_functions.RewardFunction:
  """Returns a callable reward function for being above the red block."""
  return reward_functions.DistanceReward(
      key_a=_PINCH_POSE,
      key_b=_RED_OBJECT_POSE,
      shaping_tolerance=0.05,
      offset=np.array((0., 0., _HOVER_OFFSET)),
      position_tolerance=np.array((1., 1., 0.03)))  # Anywhere horizontally.


class SparseStack(object):
  """Sparse stack reward.

  Checks that the two objects being within _X_Y_CLOSE
  of each other in the x-y plane (no constraint on z-distance). Also checks
  that the object are not in contact with the robot, to ensure the robot is
  holding the objects in place.
  """

  def __init__(self,
               top_object: prop.Prop,
               bottom_object: prop.Prop,
               get_physics_fn: Callable[[], mjcf.Physics]):
    """Initializes the reward.

    Args:
      top_object: Composer entity of the top object (red).
      bottom_object: Composer entity of the bottom object (blue).
      get_physics_fn: Callable that returns the current mjc physics from the
        environment.
    """
    self._get_physics_fn = get_physics_fn
    self._top_object = top_object
    self._bottom_object = bottom_object

  def _align(self, physics):
    return np.linalg.norm(
        self._top_object.get_pose(physics)[0][:2] -
        self._bottom_object.get_pose(physics)[0][:2]) < _X_Y_CLOSE

  def _ontop(self, physics):

    return (self._top_object.get_pose(physics)[0][2] -
            self._bottom_object.get_pose(physics)[0][2]) > _ONTOP

  def _pile(self, physics):
    geom = '{}/'.format(self._top_object.name)
    if physics_utils.has_ground_collision(physics, collision_geom_prefix=geom):
      return float(0.0)
    if physics_utils.has_robot_collision(physics, collision_geom_prefix=geom):
      return float(0.0)
    return float(1.0)

  def _collide(self, physics):
    collision_geom_prefix_1 = '{}/'.format(self._top_object.name)
    collision_geom_prefix_2 = '{}/'.format(self._bottom_object.name)
    return physics_utils.has_collision(physics, [collision_geom_prefix_1],
                                       [collision_geom_prefix_2])

  def __call__(self, obs: spec_utils.ObservationValue):
    del obs
    physics = self._get_physics_fn()
    if self._align(physics) and self._pile(physics) and self._collide(
        physics) and self._ontop(physics):
      return 1.0
    return 0.0


def get_sparse_reward_fn(
    top_object: prop.Prop,
    bottom_object: prop.Prop,
    get_physics_fn: Callable[[], mjcf.Physics]
) -> reward_functions.RewardFunction:
  """Sparse stacking reward for stacking two props with no robot contact.

  Args:
      top_object: The bottom object (blue).
      bottom_object: The top object (red).
      get_physics_fn: Callable that returns the current mjcf physics from the
        environment.

  Returns:
    The sparse stack reward function.
  """
  return SparseStack(
      top_object=top_object,
      bottom_object=bottom_object,
      get_physics_fn=get_physics_fn)
