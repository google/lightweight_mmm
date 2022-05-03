# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup for lightweight_mmm value package."""

from setuptools import find_packages
from setuptools import setup

install_requires = [
    "absl-py",
    "arviz",
    "dataclasses;python_version<'3.7'",
    "frozendict",
    "jax>=0.3.0",
    "jaxlib>=0.3.0",
    "matplotlib==3.3.4",
    "numpy>=1.12",
    "numpyro>=0.8.0",
    "scipy",
    "seaborn",
    "sklearn",
    "tensorflow==2.5.3"
]

setup(
    name="lightweight_mmm",
    version="0.1.2",
    description="Package for Media-Mix-Modelling",
    author="Google LLC",
    author_email="no-reply@google.com",
    license="Apache 2.0",
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3.7",
    ],
)
