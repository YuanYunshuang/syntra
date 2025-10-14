# Author: Yunshuang Yuan.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os

from setuptools import find_packages, setup

# Package metadata
NAME = "SYNTRA"
VERSION = "1.0"
DESCRIPTION = "SYNTRA: Segmenting Your Notions Through Retrieval-Augmentation"
URL = "https://github.com/YuanYunshuang/snytra.git"
AUTHOR = "Yunshuang Yuan"
AUTHOR_EMAIL = "yunshuang.yuan@ikg.uni-hannover.de"
LICENSE = "Apache 2.0"

# Read the contents of README file
with open("README.md", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

# Required dependencies
REQUIRED_PACKAGES = [
    "torch==2.5.1+cu118",
    "torchvision==0.20.1",
    "numpy==1.26.4",
    "tqdm>=4.67.1",
    "geotiff>=0.2.10",
    "xformers>=0.0.29",
    "pillow>=11.0.0",
    "imagecodecs>=2025.3.30",
    "hydra-core>=1.3.2",
    "matplotlib>=3.10.6",
    "iopath>=0.1.10",
    "tensordict>=0.10.0",
    "pandas>=2.3.3",
    "tensorboard>=2.20.0",
    "fvcore>=0.1.5.post20221221",
]

# EXTRA_PACKAGES = {
#     "dev": [
#     ],
# }


# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    packages=find_packages(exclude="notebooks"),
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES,
    # extras_require=EXTRA_PACKAGES,
    python_requires=">=3.10.0",
)
