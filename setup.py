from setuptools import find_packages, setup

with open("requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")

setup(
    name='sslcd',
    install_requires=install_requires,
    packages=find_packages(),
    version='0.1',
    description='Self-supervised representation learning methods (MoCo and DeepCluster) for cloud detection using Sentinel-2 images',
    author='Yawogan Jean Eudes Gbodjo',
    license='MIT'
)