from setuptools import setup

setup(
    name='stylize-datasets',
    version='0.1.0',
    packages=[''],
    url='https://github.com/bethgelab/stylize-datasets',
    license='MIT License',
    author='',
    author_email='',
    description='This repository contains code for stylizing arbitrary image datasets using AdaIN. The code is a generalization of Robert Geirhos\' Stylized-ImageNet code, which is tailored to stylizing ImageNet. Everything in this repository is based on naoto0804\'s pytorch-AdaIN implementation.  Given an image dataset, the script creates the specified number of stylized versions of every image while keeping the directory structure and naming scheme intact (usefull for existing data loaders or if directory names include class annotations).'
)
