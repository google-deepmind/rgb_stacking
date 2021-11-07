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

"""This file contains various helper functions for finding mujoco collisions.
"""

import itertools

DEFAULT_OBJECT_COLLISION_MARGIN = 0.0002
DEFAULT_COLLISION_MARGIN = 1e-8

OBJECT_GEOM_PREFIXES = ['rgb']
GROUND_GEOM_PREFIXES = ['work_surface', 'ground']
ROBOT_GEOM_PREFIXES = ['robot']


def has_object_collision(physics, collision_geom_prefix,
                         margin=DEFAULT_OBJECT_COLLISION_MARGIN):
  """Check for collisions between geoms and objects."""
  return has_collision(
      physics=physics,
      collision_geom_prefix_1=[collision_geom_prefix],
      collision_geom_prefix_2=OBJECT_GEOM_PREFIXES,
      margin=margin)


def has_ground_collision(physics, collision_geom_prefix,
                         margin=DEFAULT_COLLISION_MARGIN):
  """Check for collisions between geoms and the ground."""
  return has_collision(
      physics=physics,
      collision_geom_prefix_1=[collision_geom_prefix],
      collision_geom_prefix_2=GROUND_GEOM_PREFIXES,
      margin=margin)


def has_robot_collision(physics, collision_geom_prefix,
                        margin=DEFAULT_COLLISION_MARGIN):
  """Check for collisions between geoms and the robot."""
  return has_collision(
      physics=physics,
      collision_geom_prefix_1=[collision_geom_prefix],
      collision_geom_prefix_2=ROBOT_GEOM_PREFIXES,
      margin=margin)


def has_collision(physics, collision_geom_prefix_1, collision_geom_prefix_2,
                  margin=DEFAULT_COLLISION_MARGIN):
  """Check for collisions between geoms."""
  for contact in physics.data.contact:
    if contact.dist > margin:
      continue
    geom1_name = physics.model.id2name(contact.geom1, 'geom')
    geom2_name = physics.model.id2name(contact.geom2, 'geom')
    for pair in itertools.product(
        collision_geom_prefix_1, collision_geom_prefix_2):
      if ((geom1_name.startswith(pair[0]) and
           geom2_name.startswith(pair[1])) or
          (geom2_name.startswith(pair[0]) and
           geom1_name.startswith(pair[1]))):
        return True
  return False
