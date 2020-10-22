# encoding: utf-8
from setuptools import setup, find_packages
import sys

if (sys.version_info > (3, 0)):
    install_requires = ["scipy", "numpy"]
else:
    install_requires = ["scipy<=1.2.3", "numpy<=1.16.6"] # Last versions of scipy and numpy that support python 2.7


with open("README.md", "r") as f:
    long_description = f.read()




setup(
    name="blackboxfunctions",
    version="0.0.1",
    packages=find_packages(),
    author="Paul Knysh",
    author_email="paul.knysh@gmail.com",
    license="MIT",
    description="A Python module for parallel optimization of expensive black-box functions.",
    include_package_data=True,
    url="https://github.com/paulknysh/blackbox",
    classifiers=["Programming Language :: Python", "Programming Language :: Python :: 3", "Programming Language :: Python :: 2",],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
)

def run(self):
    __builtins__.__NUMPY_SETUP__ = False
    import numpy