from setuptools import setup

setup(
    name="deepfake-detection",
    version="0.2",
    install_requires=[
        'tensorflow-cpu==2.12.0',
        'opencv-python-headless==4.6.0.66',
    ],
)
