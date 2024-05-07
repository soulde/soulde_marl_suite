from setuptools import setup, find_packages

setup(
    name="soulde_marl_suite",
    version='1.0',
    author='soulde',
    install_requires=['numpy', 'opencv-python', 'torch', 'tqdm', 'matplotlib'],
    packages=find_packages()
)
