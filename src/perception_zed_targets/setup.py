# Copyright 2026 perception
#
# Licensed under the Apache License, Version 2.0.
"""Setup configuration for perception_zed_targets."""

import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'perception_zed_targets'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='perception',
    maintainer_email='dev@example.com',
    description='ZED RGB-D target center nodes for mission C/D.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'green_button_center_node = perception_zed_targets.green_button_center_node:main',
            'bolt_top_center_node = perception_zed_targets.bolt_top_center_node:main',
            'wheel_hole_center_node = perception_zed_targets.wheel_hole_center_node:main',
            'bolt_hole_center_node = perception_zed_targets.bolt_hole_center_node:main',
            'drill_handle_center_node = perception_zed_targets.drill_handle_center_node:main',
        ],
    },
)
