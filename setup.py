
from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='florah',
    version='0.0.0',
    description='Generate mass assembly histories with flow-based recurrent model',
    long_description=readme,
    install_requires=requirements,
    author='Tri Nguyen',
    author_email='tnguy@mit.edu',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/trivnguyen/florah',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)