from setuptools import setup, find_packages

setup(
    name='detNet',
    version='2020.6',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='Bachelor project in detector reconstruction of gamma rays using neural networks',
    long_description=open('README.md').read(),
    install_requires=['tensorflow',   #go with version 2.1 or higher just to be sure
                      'numpy',
                      'random',
                      'matplotlib',
                      'math',
                      'itertools'],
    url='https://github.com/PajterMclovin/DetNet/',
    author='Peter Halldestam',
    author_email='peterhalldestam@outlook.com'
)
