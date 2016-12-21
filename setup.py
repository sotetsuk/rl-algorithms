from setuptools import setup, find_packages

setup(
    name='rl_algorithms',
    version="0.1.0",
    description='Python implements of psuedo-codes in "Algorithms for Reinforcement Learning"',
    author='sotetsuk',
    url='https://github.com/sotetsuk/rl-algorithms',
    author_email='sotetsu.koyamada@gmail.com',
    license='MIT',
    install_requires=["numpy>=1.11.1", "gym>=0.5.6"],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: MIT License"
    ],
)
