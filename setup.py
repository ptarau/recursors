from setuptools import setup
import setuptools

import deepllm

with open('deepllm/requirements.txt') as f:
    required = f.read().splitlines()
with open("README.md", "r") as f:
    long_description = f.read()

version = deepllm.__version__
setup(name='deepllm',
      version=version,
      description='Deep, recursive, goal-driven LLM explorer',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/ptarau/recursors.git',
      author='Paul Tarau',
      author_email='paul.tarau@gmail.com',
      license='GPL-3',
      packages=setuptools.find_packages(),
      #packages=['deepllm'],
      package_data={
          'deepllm': [
              'requirements.txt',
              'tests/*.*',
              'apps/*.*',
              'demos/*.*',
              'local_llms/*.*'
          ]
      },
      include_package_data=True,
      install_requires=required,
      zip_safe=False
      )
