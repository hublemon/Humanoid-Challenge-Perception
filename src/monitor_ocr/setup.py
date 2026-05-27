from glob import glob

from setuptools import find_packages, setup

package_name = 'monitor_ocr'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=[]),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, glob('*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='dev@example.com',
    description='대시보드 모니터 OCR ROS2 노드',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'monitor_ocr_node   = monitor_ocr.monitor_ocr_node:main',
            'monitor_ocr_viewer = monitor_ocr.viewer_node:main',
        ],
    },
)
