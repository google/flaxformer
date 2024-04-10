# Copyright 2024 Google LLC.
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

"""setup.py for Flaxformer."""

import os
from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
try:
  README = open(os.path.join(here, "README.md"), encoding="utf-8").read()
except IOError:
  README = ""

install_requires = [
    "chex>=0.1.4",
    "numpy>=1.12",
    "jax>=0.2.21",
    "flax>=0.6.9",
    "aqtp>=0.1.0",
]

tests_require = [
    "absl-py",
    "immutabledict",
    "pytest",
    "tensorflow>=2.14.0",
    "tensorflow-text>=2.14.0rc0",
    "gin-config",
    "t5x @ git+https://github.com/google-research/t5x",
]

setup(
    name="flaxformer",
    version="0.8.8",
    description="Flaxformer: Transformer implementations in Flax",
    long_description="\n\n".join([README]),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="",
    author="Flaxformer team",
    author_email="noreply@google.com",
    url="https://github.com/google/flaxformer",
    packages=find_packages(),
    zip_safe=False,
    install_requires=install_requires,
    extras_require={
        "testing": tests_require,
    },
)
