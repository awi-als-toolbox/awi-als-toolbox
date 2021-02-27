# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

import os

# Get the readme
with open('README.rst') as f:
    readme = f.read()

# Get the licence
with open('LICENSE') as f:
    license = f.read()

# Get the version
mypackage_root_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(mypackage_root_dir, "awi_als_toolbox", 'VERSION')) as version_file:
    version = version_file.read().strip()

# Package requirements
with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name='awi-als-toolbox',
    version=version,
    description='python toolbox to parse and process airborne laser scanner (ALS) binary data files from the Alfred Wegener Institute (AWI) ',
    long_description=readme,
    author='Stefan Hendricks',
    author_email='stefan.hendricks@awi.de',
    url='https://github.com/shendric/awi-als-toolbox',
    license=license,
    install_requires=requirements,
    packages=find_packages(exclude=('tests', 'docs')),
    include_package_data=True
)