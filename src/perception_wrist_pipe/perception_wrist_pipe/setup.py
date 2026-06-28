# Copyright 2026 perception
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup configuration for perception_wrist_pipe."""

import os
from glob import glob

from setuptools import find_packages, setup
from setuptools.command.develop import develop as _develop

package_name = 'perception_wrist_pipe'


class ColconDevelopCommand(_develop):
    """Keep colcon editable installs compatible with newer setuptools."""

    user_options = list(_develop.user_options)
    if not any(option[0] == 'editable' for option in user_options):
        user_options.append(('editable', 'e', 'Install specified packages in editable form'))
    if not any(option[0] == 'uninstall' for option in user_options):
        user_options.append(('uninstall', 'u', 'Uninstall this source package'))
    if not any(option[0] == 'build-directory=' for option in user_options):
        user_options.append(('build-directory=', None, 'Build directory used by colcon'))

    def initialize_options(self):
        super().initialize_options()
        if not hasattr(self, 'editable'):
            self.editable = False
        if not hasattr(self, 'uninstall'):
            self.uninstall = False
        if not hasattr(self, 'build_directory'):
            self.build_directory = None
        if not hasattr(self, 'script_dir'):
            self.script_dir = None

    def finalize_options(self):
        super().finalize_options()
        install_prefix = self.install_dir.split(os.path.join('lib', 'python'), 1)[0]
        if install_prefix:
            self.script_dir = os.path.join(install_prefix, 'lib', package_name)

    def run(self):
        if getattr(self, 'uninstall', False):
            self.multi_version = True
            self.uninstall_link()
            uninstall_namespaces = getattr(self, 'uninstall_namespaces', None)
            if uninstall_namespaces is not None:
                uninstall_namespaces()
            return
        super().run()


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
    description='Wrist-camera pipe-opening top-center PoseArray node.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'wrist_pipe_top_centers_node = perception_wrist_pipe.wrist_pipe_top_centers_node:main',
            'camera_image_saver_node = perception_wrist_pipe.camera_image_saver_node:main',
        ],
    },
    cmdclass={
        'develop': ColconDevelopCommand,
    },
)
