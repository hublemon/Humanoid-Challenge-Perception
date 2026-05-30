from glob import glob

from setuptools import find_packages, setup

package_name = 'task_management'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    scripts=['scripts/tray_occupancy_node'],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='base',
    maintainer_email='base@todo.todo',
    description='Task management nodes for blue tray occupancy and OCR task lists.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'management_node = task_management.management_node:main',
        ],
    },
)
