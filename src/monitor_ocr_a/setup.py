import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'monitor_ocr_a'


def template_data_files():
    entries = [
        ('share/' + package_name + '/templates/icons', glob('templates/icons/*')),
        ('share/' + package_name + '/templates/digits', glob('templates/digits/*.png')),
    ]
    for root, _dirs, files in os.walk('templates/digits'):
        image_files = [
            os.path.join(root, name)
            for name in files
            if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        if root != 'templates/digits' and image_files:
            entries.append(('share/' + package_name + '/' + root, image_files))
    return entries


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=[]),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['best.pt']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ] + template_data_files(),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='rnrqls0326@snu.ac.kr',
    description='대시보드 모니터 OCR ROS2 노드',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'monitor_ocr_a_node   = monitor_ocr_a.monitor_ocr_node:main',
            'monitor_ocr_a_viewer = monitor_ocr_a.viewer_node:main',
        ],
    },
)
