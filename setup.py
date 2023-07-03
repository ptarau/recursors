from setuptools import setup
import setuptools

import deepllm

with open('requirements.txt') as f:
    required = f.read().splitlines()
with open("README.md", "r") as f:
    long_description = f.read()

version = deepllm.__version__
setup(name='deepllm',
      version=version,
      description='Prolog-like interpreter and tuple store',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/ptarau/recursors.git',
      author='Paul Tarau',
      author_email='paul.tarau@gmail.com',
      license='MIT',
      packages=setuptools.find_packages(),
      package_data={'deepllm': [
                               'docs/*.pdf',
                               'tests/*.py',
                               'apps/*.py',
                               'demos/*.py'
                               'local_llms/*.py'
                               ]
                    },
      include_package_data=True,
      install_requires=required,
      zip_safe=False
      )
