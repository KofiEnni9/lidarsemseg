from setuptools import setup, find_packages

setup(
    name='lidarsemseg',
    version='0.1.0',
    description='LiDAR Semantic Segmentation Package',
    author='Your Name',
    author_email='kofienninachepong@gmall.com',
    packages=find_packages(),
    install_requires=[
        'open3d',
        'numpy',
        'torch',
    ],
    package_data={
        'lidarsemseg': ['data_files/*'],
    },
    python_requires='>=3.6',
)
