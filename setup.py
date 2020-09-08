# -*- coding: utf-8 -*-

import re

from setuptools import find_packages, setup


with open("cryptomatte_masks/__init__.py") as f:
    txt = f.read()
    try:
        version = re.findall(r'^__version__ = "([^"]+)"\r?$', txt, re.M)[0]
    except IndexError:
        raise RuntimeError("Unable to determine version.")

setup(
    name="cryptomatte_masks",
    version=version,
    python_requires=">=3.6.0",
    install_requires=[
        "numpy>=1.18.0",
        "exr-info @ git+https://github.com/Synthesis-AI-Dev/exr-info.git#egg=exr-info-0.1",
        "openexr @ git+https://github.com/jamesbowman/openexrpython.git#egg=openexr-1.3.2",
    ],
    description="Extracts cryptomattes from rendered EXR files",
    packages=find_packages(),
)
