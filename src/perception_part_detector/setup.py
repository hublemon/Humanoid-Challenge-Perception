from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'perception_part_detector'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    py_modules=['detector_node'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/weights', glob('weights/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='base',
    maintainer_email='base@todo.todo',
    description='YOLO-based part detector node',
    license='MIT',
    entry_points={
        'console_scripts': [
            'detector_node = detector_node:main',
        ],
    },
)
