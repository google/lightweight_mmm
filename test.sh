#!/bin/bash
# Copyright 2023 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set -e

readonly VENV_DIR=/tmp/test_env
echo "Creating virtual environment under ${VENV_DIR}."
echo "You might want to remove this when you no longer need it."

# Install deps in a virtual env.
python -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python --version

# Install JAX.
python -m pip install --upgrade pip setuptools
python -m pip install -r requirements/requirements.txt
python -c 'import jax; print(jax.__version__)'

# Run setup.py, this installs the python dependencies
python -m pip install .

# Python test dependencies.
python -m pip install -r requirements/requirements_tests.txt

# Run tests using pytest.
python -m pytest lightweight_mmm