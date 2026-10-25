"""Installation script for the DreamWaQ Lab Isaac Lab extension."""

from pathlib import Path

import toml
from setuptools import find_packages, setup


EXTENSION_PATH = Path(__file__).resolve().parent
EXTENSION_DATA = toml.load(EXTENSION_PATH / "config" / "extension.toml")

setup(
    name="dreamwaq_lab",
    version=EXTENSION_DATA["package"]["version"],
    description=EXTENSION_DATA["package"]["description"],
    author=EXTENSION_DATA["package"]["author"],
    maintainer=EXTENSION_DATA["package"]["maintainer"],
    url=EXTENSION_DATA["package"]["repository"],
    license="Apache-2.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["psutil", "PyYAML"],
    python_requires=">=3.10",
    zip_safe=False,
)
