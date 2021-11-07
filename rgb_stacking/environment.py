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

"""Builds RGB stacking environment.

This file builds the RGB stacking environment used in the paper
"Beyond Pick-and-Place: Tackling robotic stacking of diverse shapes".
The environment is composed of a robot with a parallel gripper. In front
of the robot there is a basket containing 3 objects, one red, one green and one
blue. The goal is for the robot to stack the red object on top of the blue one.

In this specific file, we build the interface that is exposed to the agent,
namely the action spec, the observation spec along with the reward.
"""

import enum
from typing import Sequence

from dm_robotics import agentflow as af
from dm_robotics.agentflow.preprocessors import observation_transforms
from dm_robotics.agentflow.preprocessors import timestep_preprocessor as tsp
from dm_robotics.agentflow.subtasks import subtask_termination
from dm_robotics.manipulation.props.rgb_objects import rgb_object
from dm_robotics.moma import action_spaces
from dm_robotics.moma import subtask_env
from dm_robotics.moma import subtask_env_builder
from dm_robotics.moma.utils import mujoco_collisions
import numpy as np

from rgb_stacking import reward_functions
from rgb_stacking import stack_rewards
from rgb_stacking import task


# The environments provides stacked observations to the agent. The data of the
# previous n steps is stacked and provided as a single observation to the agent.
# We stack different observations a different number of times.
_OBSERVATION_STACK_DEPTH = 3
_ACTION_OBS_STACK_DEPTH = 2

# Number of steps in an episode
_MAX_STEPS = 400

# Timestep of the physics simulation.
_PHYSICS_TIMESTEP = 0.0005


_STATE_OBSERVATIONS = (
    'action/environment',
    'gripper/grasp',
    'gripper/joints/angle',
    'gripper/joints/velocity',
    'rgb30_blue/abs_pose',
    'rgb30_blue/to_pinch',
    'rgb30_green/abs_pose',
    'rgb30_green/to_pinch',
    'rgb30_red/abs_pose',
    'rgb30_red/to_pinch',
    'sawyer/joints/angle',
    'sawyer/joints/torque',
    'sawyer/joints/velocity',
    'sawyer/pinch/pose',
    'sawyer/tcp/pose',
    'sawyer/tcp/velocity',
    'wrist/force',
    'wrist/torque'
)

_VISION_OBSERVATIONS = (
    'basket_back_left/pixels',
    'basket_front_left/pixels',
    'basket_front_right/pixels'
)

# For interactive imitation learning, the vision-based policy used the
# following proprioception and images observations from the pair of front
# cameras given by the simulated environment.
_INTERACTIVE_IMITATION_LEARNING_OBSERVATIONS = (
    'sawyer/joints/angle',
    'gripper/joints/angle',
    'sawyer/pinch/pose',
    'sawyer/tcp/pose',
    'basket_front_left/pixels',
    'basket_front_right/pixels'
)


# For the one-step offline policy improvement from real data, the vision-based
# policy used the following proprioception and images observations from the pair
# of front cameras given by the real environment.
_OFFLINE_POLICY_IMPROVEMENT_OBSERVATIONS = [
    'sawyer/joints/angle',
    'sawyer/joints/velocity',
    'gripper/grasp',
    'gripper/joints/angle',
    'gripper/joints/velocity',
    'sawyer/pinch/pose',
    'basket_front_left/pixels',
    'basket_front_right/pixels',]


class ObservationSet(int, enum.Enum):
  """Different possible set of observations that can be exposed."""

  _observations: Sequence[str]

  def __new__(cls, value: int, observations: Sequence[str]):
    obj = int.__new__(cls, value)
    obj._value_ = value
    obj._observations = observations
    return obj

  @property
  def observations(self):
    return self._observations

  STATE_ONLY = (0, _STATE_OBSERVATIONS)
  VISION_ONLY = (1, _VISION_OBSERVATIONS)
  ALL = (2, _STATE_OBSERVATIONS + _VISION_OBSERVATIONS)
  INTERACTIVE_IMITATION_LEARNING = (
      3, _INTERACTIVE_IMITATION_LEARNING_OBSERVATIONS)
  OFFLINE_POLICY_IMPROVEMENT = (4, _OFFLINE_POLICY_IMPROVEMENT_OBSERVATIONS)


def rgb_stacking(
    object_triplet: str = 'rgb_test_random',
    observation_set: ObservationSet = ObservationSet.STATE_ONLY,
    use_sparse_reward: bool = False
) -> subtask_env.SubTaskEnvironment:
  """Returns the environment.

  The relevant groups can be found here:
  https://github.com/deepmind/robotics/blob/main/py/manipulation/props/rgb_objects/rgb_object.py

  The valid object triplets can be found under PROP_TRIPLETS in the file.

  Args:
    object_triplet: Triplet of RGB objects to use in the environment.
    observation_set: Set of observations that en environment should expose.
    use_sparse_reward: If true will use sparse reward, which is 1 if the objects
      stacked and not touching the robot, and 0 otherwise.
  """

  red_id, green_id, blue_id = rgb_object.PROP_TRIPLETS[object_triplet][1]
  rgb_task = task.rgb_task(red_id, green_id, blue_id)
  rgb_task.physics_timestep = _PHYSICS_TIMESTEP

  # To speed up simulation we ensure that mujoco will no check contact between
  # geoms that cannot collide.
  mujoco_collisions.exclude_bodies_based_on_contype_conaffinity(
      rgb_task.root_entity.mjcf_model)

  # Build the agent flow subtask. This is where the task logic is defined,
  # observations, and rewards.
  env_builder = subtask_env_builder.SubtaskEnvBuilder()
  env_builder.set_task(rgb_task)
  task_env = env_builder.build_base_env()

  # Define the action space, this is used to expose the actuators used in the
  # base task.
  effectors_action_spec = rgb_task.effectors_action_spec(
      physics=task_env.physics)
  robot_action_spaces = []
  for rbt in rgb_task.robots:
    arm_action_space = action_spaces.ArmJointActionSpace(
        af.prefix_slicer(effectors_action_spec, rbt.arm_effector.prefix))
    gripper_action_space = action_spaces.GripperActionSpace(
        af.prefix_slicer(effectors_action_spec, rbt.gripper_effector.prefix))
    robot_action_spaces.extend([arm_action_space, gripper_action_space])

  composite_action_space = af.CompositeActionSpace(
      robot_action_spaces)
  env_builder.set_action_space(composite_action_space)

  # Cast all the floating point observations to float32.
  env_builder.add_preprocessor(
      observation_transforms.DowncastFloatPreprocessor(np.float32))

  # Concatenate the TCP and wrist site observations.
  env_builder.add_preprocessor(observation_transforms.MergeObservations(
      obs_to_merge=['robot0_tcp_pos', 'robot0_tcp_quat'],
      new_obs='robot0_tcp_pose'))
  env_builder.add_preprocessor(observation_transforms.MergeObservations(
      obs_to_merge=['robot0_wrist_site_pos', 'robot0_wrist_site_quat'],
      new_obs='robot0_wrist_site_pose'))

  # Add in observations to measure the distance from the TCP to the objects.
  for color in ('red', 'green', 'blue'):
    env_builder.add_preprocessor(observation_transforms.AddObservation(
        obs_name=f'{color}_to_pinch',
        obs_callable=_distance_delta_obs(
            f'rgb_object_{color}_pose', 'robot0_tcp_pose')))

  # Concatenate the action sent to the robot joints and the gripper actuator.
  env_builder.add_preprocessor(observation_transforms.MergeObservations(
      obs_to_merge=['robot0_arm_joint_previous_action',
                    'robot0_gripper_previous_action'],
      new_obs='robot0_previous_action'))

  # Mapping of observation names to match the observation names in the stored
  # data.
  obs_mapping = {
      'robot0_arm_joint_pos': 'sawyer/joints/angle',
      'robot0_arm_joint_vel': 'sawyer/joints/velocity',
      'robot0_arm_joint_torques': 'sawyer/joints/torque',
      'robot0_tcp_pose': 'sawyer/pinch/pose',
      'robot0_wrist_site_pose': 'sawyer/tcp/pose',
      'robot0_wrist_site_vel_world': 'sawyer/tcp/velocity',
      'robot0_gripper_pos': 'gripper/joints/angle',
      'robot0_gripper_vel': 'gripper/joints/velocity',
      'robot0_gripper_grasp': 'gripper/grasp',
      'robot0_wrist_force': 'wrist/force',
      'robot0_wrist_torque': 'wrist/torque',
      'rgb_object_red_pose': 'rgb30_red/abs_pose',
      'rgb_object_green_pose': 'rgb30_green/abs_pose',
      'rgb_object_blue_pose': 'rgb30_blue/abs_pose',
      'basket_back_left_rgb_img': 'basket_back_left/pixels',
      'basket_front_left_rgb_img': 'basket_front_left/pixels',
      'basket_front_right_rgb_img': 'basket_front_right/pixels',
      'red_to_pinch': 'rgb30_red/to_pinch',
      'blue_to_pinch': 'rgb30_blue/to_pinch',
      'green_to_pinch': 'rgb30_green/to_pinch',
      'robot0_previous_action': 'action/environment',
      }

  # Create different subsets of observations.
  action_obs = {'action/environment'}

  # These observations only have a single floating point value instead of an
  # array.
  single_value_obs = {'gripper/joints/angle',
                      'gripper/joints/velocity',
                      'gripper/grasp'}

  # Rename observations.
  env_builder.add_preprocessor(observation_transforms.RenameObservations(
      obs_mapping, raise_on_missing=False))

  if use_sparse_reward:
    reward_fn = stack_rewards.get_sparse_reward_fn(
        top_object=rgb_task.props[0],
        bottom_object=rgb_task.props[2],
        get_physics_fn=lambda: task_env.physics)
  else:
    reward_fn = stack_rewards.get_shaped_stacking_reward()
  env_builder.add_preprocessor(reward_functions.RewardPreprocessor(reward_fn))

  # We concatenate several observations from consecutive timesteps. Depending
  # on the observations, we will concatenate a different number of observations.
  # - Most observations are stacked 3 times
  # - Camera observations are not stacked.
  # - The action observation is stacked twice.
  # - When stacking three scalar (i.e. numpy array of shape (1,)) observations,
  #   we do not add a leading dimension, so the final shape is (3,).
  env_builder.add_preprocessor(
      observation_transforms.StackObservations(
          obs_to_stack=list(
              set(_STATE_OBSERVATIONS) - action_obs - single_value_obs),
          stack_depth=_OBSERVATION_STACK_DEPTH,
          add_leading_dim=True))
  env_builder.add_preprocessor(
      observation_transforms.StackObservations(
          obs_to_stack=list(single_value_obs),
          stack_depth=_OBSERVATION_STACK_DEPTH,
          add_leading_dim=False))
  env_builder.add_preprocessor(
      observation_transforms.StackObservations(
          obs_to_stack=list(action_obs),
          stack_depth=_ACTION_OBS_STACK_DEPTH,
          add_leading_dim=True))

  # Only keep the obseravtions that we want to expose to the agent.
  env_builder.add_preprocessor(observation_transforms.RetainObservations(
      observation_set.observations, raise_on_missing=False))

  # End episodes after 400 steps.
  env_builder.add_preprocessor(
      subtask_termination.MaxStepsTermination(_MAX_STEPS))

  return env_builder.build()


def _distance_delta_obs(key1: str, key2: str):
  """Returns a callable that returns the difference between two observations."""
  def util(timestep: tsp.PreprocessorTimestep) -> np.ndarray:
    return timestep.observation[key1] - timestep.observation[key2]
  return util

