import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'perception_2d_to_pcd_wrist'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='perception',
    maintainer_email='dev@example.com',
    description='Wrist (RealSense) 2D detections -> 3D pose / PointCloud in base_link, '
                'with depth->color re-projection for unaligned RGB-D.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'wrist_projection_node = perception_2d_to_pcd_wrist.wrist_projection_node:main',
            'wrist_pointcloud_node = perception_2d_to_pcd_wrist.wrist_pointcloud_node:main',
            'wrist_grasp_pcd_node = perception_2d_to_pcd_wrist.wrist_grasp_pcd_node:main',
        ],
    },
)
