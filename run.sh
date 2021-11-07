#!/bin/bash
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

# Fail on any error.
set -e

# Display commands being run.
set -x

TMP_DIR=`mktemp -d`

python3 -m venv "${TMP_DIR}/rgb_stacking"
source "${TMP_DIR}/rgb_stacking/bin/activate"

# Install dependencies.
pip install --upgrade -r requirements.txt

# Run the visualization of the environment.
python -m rgb_stacking.main

# Clean up.
rm -r ${TMP_DIR}
