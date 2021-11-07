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

"""Script to run a viewer to visualize the rgb stacking environment."""

from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import cv2
from dm_control import viewer
from dm_robotics.manipulation.props.rgb_objects import rgb_object
from dm_robotics.moma import subtask_env
import numpy as np

from rgb_stacking import environment
from rgb_stacking.utils import policy_loading

_TEST_OBJECT_TRIPLETS = tuple(rgb_object.PROP_TRIPLETS_TEST.keys())

_ALL_OBJECT_TRIPLETS = _TEST_OBJECT_TRIPLETS + (
    'rgb_train_random',
    'rgb_test_random',
    )

_POLICY_DIR = ('assets/saved_models')

_POLICY_PATHS = {
    k: f'{_POLICY_DIR}/mpo_state_{k}' for k in _TEST_OBJECT_TRIPLETS
}

_OBJECT_TRIPLET = flags.DEFINE_enum(
    'object_triplet', 'rgb_test_random', _ALL_OBJECT_TRIPLETS,
    'Triplet of RGB objects to use in the environment.')
_POLICY_OBJECT_TRIPLET = flags.DEFINE_enum(
    'policy_object_triplet', None, _TEST_OBJECT_TRIPLETS,
    'Optional test triplet name indicating to load a policy that was trained on'
    ' this triplet.')
_LAUNCH_VIEWER = flags.DEFINE_bool(
    'launch_viewer', True,
    'Optional boolean. If True, will launch the dm_control viewer. If False'
    ' will load the policy, run it and save a recording of it as an .mp4.')


def run_episode_and_render(
    env: subtask_env.SubTaskEnvironment,
    policy: policy_loading.Policy
) -> Sequence[np.ndarray]:
  """Saves a gif of the policy running against the environment."""
  rendered_images = []
  logging.info('Starting the rendering of the policy, this might take some'
               ' time...')
  state = policy.initial_state()
  timestep = env.reset()
  rendered_images.append(env.physics.render(camera_id='main_camera'))
  while not timestep.last():
    (action, _), state = policy.step(timestep, state)
    timestep = env.step(action)
    rendered_images.append(env.physics.render(camera_id='main_camera'))
  logging.info('Done rendering!')
  return rendered_images


def main(argv: Sequence[str]) -> None:

  del argv

  if not _LAUNCH_VIEWER.value and _POLICY_OBJECT_TRIPLET.value is None:
    raise ValueError('To record a video, a policy must be given.')

  # Load the rgb stacking environment.
  with environment.rgb_stacking(object_triplet=_OBJECT_TRIPLET.value) as env:

    # Optionally load a policy trained on one of these environments.
    if _POLICY_OBJECT_TRIPLET.value is not None:
      policy_path = _POLICY_PATHS[_POLICY_OBJECT_TRIPLET.value]
      policy = policy_loading.policy_from_path(policy_path)
    else:
      policy = None

    if _LAUNCH_VIEWER.value:
      # The viewer requires a callable as a policy.
      if policy is not None:
        policy = policy_loading.StatefulPolicyCallable(policy)
      viewer.launch(env, policy=policy)
    else:

      # Render the episode.
      rendered_episode = run_episode_and_render(env, policy)

      # Save as mp4 video in current directory.
      height, width, _ = rendered_episode[0].shape
      out = cv2.VideoWriter(
          './rendered_policy.mp4',
          cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
          1.0 / env.task.control_timestep, (width, height))
      for image in rendered_episode:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out.write(image)
      out.release()

if __name__ == '__main__':
  app.run(main)
