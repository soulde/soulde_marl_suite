from setuptools import setup, find_packages

setup(
    name="easier_mj",
    version='1.0',
    author='soulde',
    install_requires=['numpy', 'opencv-python', 'torch', 'tqdm', 'matplotlib', 'mujoco'],
    packages=find_packages()
)
