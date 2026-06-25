from setuptools import find_packages, setup

package_name = 'perception_part_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='base',
    maintainer_email='base@todo.todo',
    description='Part detector node',
    license='MIT',
)
