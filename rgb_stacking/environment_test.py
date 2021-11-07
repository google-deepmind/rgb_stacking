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

"""Tests environment.py."""

from absl.testing import absltest
from dm_env import test_utils

from rgb_stacking import environment


class EnvironmentTest(test_utils.EnvironmentTestMixin, absltest.TestCase):

  def make_object_under_test(self):
    return environment.rgb_stacking(
        object_triplet='rgb_test_triplet1',
        observation_set=environment.ObservationSet.ALL)


if __name__ == '__main__':
  absltest.main()
