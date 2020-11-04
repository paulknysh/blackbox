# -*- coding: utf-8 -*-

import os
import sys
from setuptools import setup



def read_long_description(filename):
    path = os.path.realpath(__file__)
    path, _ = os.path.split(path)
    with open(os.path.join(path, filename), 'r') as f:
        long_description = f.read()
    return long_description

def get_install_requirements():
    if (sys.version_info > (3, 0)):
        return ['scipy', 'numpy']
    else:
        return ['scipy<=1.2.3', 'numpy<=1.16.6'] # Last versions of scipy and numpy that support python 2.7

def get_develop_requirements():
    return get_install_requirements() + ["pytest"]

setup(
    name='blackboxfunctions',
    version='0.0.1',
    
    # Descriptions
    description='A Python module for parallel optimization of expensive black-box functions.',
    long_description=read_long_description('README.md'),
    long_description_content_type='text/markdown',
    
    # Urls
    url='https://github.com/paulknysh/blackbox',
    download_url='https://github.com/paulknysh/blackbox',

    # Author details
    author='Paul Knysh',
    author_email='paul.knysh@gmail.com',
    license='MIT',
   
    # Classifiers and keywords https://pypi.org/classifiers/
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python', 
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries'
    ],
    keywords='optimization, black-box function, Latin hypercube, response surface, parallel computing',

    # Packaging
    packages=['blackboxfunctions'],
    include_package_data=True,
    install_requires=get_install_requirements(),
    test_suite="pytest",
    extras_require={
        'develop': get_develop_requirements()
    }
)