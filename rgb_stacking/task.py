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

"""A module for constructing the RGB stacking task.

This file builds the composer task containing a single sawyer robot facing
3 objects: a red, a green and a blue one.

We define:
 - All the simulation objects, robot, basket, objects.
 - The sensors to measure the state of the environment.
 - The effector to control the robot.
 - The initialization logic.

On top of this we can build a MoMa subtask environment. In this subtask
environment we will decide what the reward will be and what observations are
exposed. Thus allowing us to change the goal without changing this environment.
"""
from typing import Sequence

from dm_control import composer
from dm_control.composer.variation import distributions
from dm_control.composer.variation import rotations
from dm_robotics.geometry import pose_distribution
from dm_robotics.manipulation.props.rgb_objects import rgb_object
from dm_robotics.manipulation.standard_cell import rgb_basket
from dm_robotics.moma import base_task
from dm_robotics.moma import entity_initializer
from dm_robotics.moma import prop
from dm_robotics.moma import robot as moma_robot
from dm_robotics.moma.effectors import arm_effector as arm_effector_module
from dm_robotics.moma.effectors import cartesian_4d_velocity_effector
from dm_robotics.moma.effectors import cartesian_6d_velocity_effector
from dm_robotics.moma.effectors import default_gripper_effector
from dm_robotics.moma.effectors import min_max_effector
from dm_robotics.moma.models.arenas import empty
from dm_robotics.moma.models.end_effectors.robot_hands import robotiq_2f85
from dm_robotics.moma.models.end_effectors.wrist_sensors import robotiq_fts300
from dm_robotics.moma.models.robots.robot_arms import sawyer
from dm_robotics.moma.models.robots.robot_arms import sawyer_constants
from dm_robotics.moma.sensors import action_sensor
from dm_robotics.moma.sensors import camera_sensor
from dm_robotics.moma.sensors import prop_pose_sensor
from dm_robotics.moma.sensors import robot_arm_sensor
from dm_robotics.moma.sensors import robot_tcp_sensor
from dm_robotics.moma.sensors import robot_wrist_ft_sensor
from dm_robotics.moma.sensors import robotiq_gripper_sensor
from dm_robotics.moma.sensors import site_sensor
import numpy as np


# Margin from the joint limits at which to stop the Z rotation when using 4D
# control. Values chosen to match the existing PyRobot-based environments.
WRIST_RANGE = (-np.pi / 2, np.pi / 2)


# Position of the basket relative to the attachment point of the robot.
_DEFAULT_BASKET_CENTER = (0.6, 0.)
DEFAULT_BASKET_HEIGHT = 0.0498
_BASKET_ORIGIN = _DEFAULT_BASKET_CENTER + (DEFAULT_BASKET_HEIGHT,)
WORKSPACE_CENTER = np.array(_DEFAULT_BASKET_CENTER + (0.1698,))
WORKSPACE_SIZE = np.array([0.25, 0.25, 0.2])
# Maximum linear and angular velocity of the robot's TCP.
_MAX_LIN_VEL = 0.07
_MAX_ANG_VEL = 1.0

# Limits of the distributions used to sample initial positions (X, Y, Z in [m])
# for props in the basket.
_PROP_MIN_POSITION_BOUNDS = [0.50, -0.10, 0.12]
_PROP_MAX_POSITION_BOUNDS = [0.70, 0.10, 0.12]

# Limits of the distributions used to sample initial position for TCP.
_TCP_MIN_POSE_BOUNDS = [0.5, -0.14, 0.22, np.pi, 0, -np.pi / 4]
_TCP_MAX_POSE_BOUNDS = [0.7, 0.14, 0.43, np.pi, 0, np.pi / 4]

# Control timestep exposed to the agent.
_CONTROL_TIMESTEP = 0.05

# Joint state used for the nullspace.
_NULLSPACE_JOINT_STATE = [
    0.0, -0.5186220703125, -0.529384765625, 1.220857421875, 0.40857421875,
    1.07831640625, 0.0]

# Joint velocity magnitude limits from the Sawyer URDF.
_JOINT_VEL_LIMITS = sawyer_constants.VELOCITY_LIMITS['max']

# Identifier for the cameras. The key is the name used for the MoMa camera
# sensor and the value corresponds to the identifier of that camera in the
# mjcf model.
_CAMERA_IDENTIFIERS = {'basket_back_left': 'base/basket_back_left',
                       'basket_front_left': 'base/basket_front_left',
                       'basket_front_right': 'base/basket_front_right'}

# Configuration of the MuJoCo cameras.
_CAMERA_CONFIG = camera_sensor.CameraConfig(
    width=128,
    height=128,
    fovy=30.,
    has_rgb=True,
    has_depth=False,
)


def rgb_task(red_obj_id: str,
             green_obj_id: str,
             blue_obj_id: str) -> base_task.BaseTask:
  """Builds a BaseTask and all dependencies.

  Args:
    red_obj_id: The RGB object ID that corresponds to the red object. More
    information on this can be found in the RGB Objects file.
    green_obj_id: See `red_obj_id`
    blue_obj_id: See `red_obj_id`

  Returns:
     The modular manipulation (MoMa) base task for the RGB stacking environment.
     A robot is placed in front of a basket containing 3 objects: a red, a green
     and blue one.
  """

  # Build the composer scene.
  arena = _arena()
  _workspace(arena)

  robot = _sawyer_robot(robot_name='robot0')
  arena.attach(robot.arm)

  # We add a camera with a good point of view for capturing videos.
  pos = '1.4 0.0 0.45'
  quat = '0.541  0.455  0.456  0.541'
  name = 'main_camera'
  fovy = '45'
  arena.mjcf_model.worldbody.add(
      'camera', name=name, pos=pos, quat=quat, fovy=fovy)

  props = _props(red_obj_id, green_obj_id, blue_obj_id)
  for p in props:
    frame = arena.add_free_entity(p)
    p.set_freejoint(frame.freejoint)

  # Add in the MoMa sensor to get observations from the environment.
  extra_sensors = prop_pose_sensor.build_prop_pose_sensors(props)
  camera_configurations = {
      name: _CAMERA_CONFIG for name in _CAMERA_IDENTIFIERS.keys()}
  extra_sensors.extend(
      camera_sensor.build_camera_sensors(
          camera_configurations, arena.mjcf_model, _CAMERA_IDENTIFIERS))

  # Initializers to place the TCP and the props in the basket.
  dynamic_initializer = entity_initializer.TaskEntitiesInitializer(
      [_gripper_initializer(robot), _prop_initializers(props)])

  moma_task = base_task.BaseTask(
      task_name='rgb_stacking',
      arena=arena,
      robots=[robot],
      props=props,
      extra_sensors=extra_sensors,
      extra_effectors=[],
      scene_initializer=lambda _: None,
      episode_initializer=dynamic_initializer,
      control_timestep=_CONTROL_TIMESTEP)
  return moma_task


def _workspace(arena: composer.Arena) -> rgb_basket.RGBBasket:
  """Returns the basket used in the rgb stacking environment."""
  workspace = rgb_basket.RGBBasket()
  attachment_site = arena.mjcf_model.worldbody.add(
      'site', pos=_BASKET_ORIGIN, rgba='0 0 0 0', size='0.01')
  arena.attach(workspace, attachment_site)
  return workspace


def _gripper_initializer(
    robot: moma_robot.Robot) -> entity_initializer.PoseInitializer:
  """Populates components with gripper initializers."""

  gripper_pose_dist = pose_distribution.UniformPoseDistribution(
      min_pose_bounds=_TCP_MIN_POSE_BOUNDS,
      max_pose_bounds=_TCP_MAX_POSE_BOUNDS)
  return entity_initializer.PoseInitializer(robot.position_gripper,
                                            gripper_pose_dist.sample_pose)


def _prop_initializers(
    props: Sequence[prop.Prop]) -> entity_initializer.PropPlacer:
  """Populates components with prop pose initializers."""
  prop_position = distributions.Uniform(_PROP_MIN_POSITION_BOUNDS,
                                        _PROP_MAX_POSITION_BOUNDS)
  prop_quaternion = rotations.UniformQuaternion()

  return entity_initializer.PropPlacer(
      props=props,
      position=prop_position,
      quaternion=prop_quaternion,
      settle_physics=True)


def _arena() -> composer.Arena:
  """Builds an arena Entity."""
  arena = empty.Arena()
  arena.mjcf_model.size.nconmax = 5000
  arena.mjcf_model.size.njmax = 5000

  return arena


def _sawyer_robot(robot_name: str) -> moma_robot.Robot:
  """Returns a Sawyer robot with all the sensors and effectors."""

  arm = sawyer.Sawyer(
      name=robot_name, actuation=sawyer_constants.Actuation.INTEGRATED_VELOCITY)

  gripper = robotiq_2f85.Robotiq2F85()

  wrist_ft = robotiq_fts300.RobotiqFTS300()

  wrist_cameras = []

  # Compose the robot after its model components are constructed. This should
  # usually be done early on as some Effectors (and possibly Sensors) can only
  # be constructed after the robot components have been composed.
  moma_robot.standard_compose(
      arm=arm, gripper=gripper, wrist_ft=wrist_ft, wrist_cameras=wrist_cameras)

  # We need to measure the last action sent to the robot and the gripper.
  arm_effector, arm_action_sensor = action_sensor.create_sensed_effector(
      arm_effector_module.ArmEffector(
          arm=arm, action_range_override=None, robot_name=robot_name))

  # Effector used for the gripper. The gripper is controlled by applying the
  # min or max command, this allows the agent to quicky learn how to grasp
  # instead of learning how to close the gripper first.
  gripper_effector, gripper_action_sensor = action_sensor.create_sensed_effector(
      default_gripper_effector.DefaultGripperEffector(gripper, robot_name))

  # Enable bang bang control for the gripper, this allows the agent to close and
  # open the gripper faster.
  gripper_effector = min_max_effector.MinMaxEffector(
      base_effector=gripper_effector)

  # Build the 4D cartesian controller, we use a 6D cartesian effector under the
  # hood.
  effector_model = cartesian_6d_velocity_effector.ModelParams(
      element=arm.wrist_site, joints=arm.joints)
  effector_control = cartesian_6d_velocity_effector.ControlParams(
      control_timestep_seconds=_CONTROL_TIMESTEP,
      max_lin_vel=_MAX_LIN_VEL,
      max_rot_vel=_MAX_ANG_VEL,
      joint_velocity_limits=np.array(_JOINT_VEL_LIMITS),
      nullspace_gain=0.025,
      nullspace_joint_position_reference=np.array(_NULLSPACE_JOINT_STATE),
      regularization_weight=1e-2,
      enable_joint_position_limits=True,
      minimum_distance_from_joint_position_limit=0.01,
      joint_position_limit_velocity_scale=0.95,
      max_cartesian_velocity_control_iterations=300,
      max_nullspace_control_iterations=300)

  # Don't activate collision avoidance because we are restricted to the virtual
  # workspace in the center of the basket.
  cart_effector_6d = cartesian_6d_velocity_effector.Cartesian6dVelocityEffector(
      robot_name=robot_name,
      joint_velocity_effector=arm_effector,
      model_params=effector_model,
      control_params=effector_control)
  cart_effector_4d = cartesian_4d_velocity_effector.Cartesian4dVelocityEffector(
      effector_6d=cart_effector_6d,
      element=arm.wrist_site,
      effector_prefix=f'{robot_name}_cart_4d_vel')

  # Constrain the workspace of the robot.
  cart_effector_4d = cartesian_4d_velocity_effector.limit_to_workspace(
      cartesian_effector=cart_effector_4d,
      element=gripper.tool_center_point,
      min_workspace_limits=WORKSPACE_CENTER - WORKSPACE_SIZE / 2,
      max_workspace_limits=WORKSPACE_CENTER + WORKSPACE_SIZE / 2,
      wrist_joint=arm.joints[-1],
      wrist_limits=WRIST_RANGE,
      reverse_wrist_range=True)

  robot_sensors = []

  # Sensor for the joint states (torques, velocities and angles).
  robot_sensors.append(robot_arm_sensor.RobotArmSensor(
      arm=arm, name=f'{robot_name}_arm', have_torque_sensors=True))

  # Sensor for the cartesian pose of the tcp site.
  robot_sensors.append(robot_tcp_sensor.RobotTCPSensor(
      gripper=gripper, name=robot_name))

  # Sensor for cartesian pose of the wrist site.
  robot_sensors.append(site_sensor.SiteSensor(
      site=arm.wrist_site, name=f'{robot_name}_wrist_site'))

  # Sensor to measure the state of the gripper (position, velocity and grasp).
  robot_sensors.append(robotiq_gripper_sensor.RobotiqGripperSensor(
      gripper=gripper, name=f'{robot_name}_gripper'))

  # Sensor for the wrench measured at the wrist sensor.
  robot_sensors.append(robot_wrist_ft_sensor.RobotWristFTSensor(
      wrist_ft_sensor=wrist_ft, name=f'{robot_name}_wrist'))

  # Sensors to measure the last action sent to the arm joints and the gripper
  # actuator.
  robot_sensors.extend([arm_action_sensor, gripper_action_sensor])

  return moma_robot.StandardRobot(
      arm=arm,
      arm_base_site_name='base_site',
      gripper=gripper,
      robot_sensors=robot_sensors,
      wrist_cameras=wrist_cameras,
      arm_effector=cart_effector_4d,
      gripper_effector=gripper_effector,
      wrist_ft=wrist_ft,
      name=robot_name)


def _props(red: str, green: str, blue: str) -> Sequence[prop.Prop]:
  """Build task props."""
  objects = ((red, 'red'), (green, 'green'), (blue, 'blue'))
  color_set = [
      [1, 0, 0, 1],
      [0, 1, 0, 1],
      [0, 0, 1, 1],
  ]
  props = []
  for i, (obj_id, color) in enumerate(objects):
    p = rgb_object.RgbObjectProp(
        obj_id=obj_id, color=color_set[i], name=f'rgb_object_{color}')
    p = prop.WrapperProp(wrapped_entity=p, name=f'rgb_object_{color}')
    props.append(p)

  return props
