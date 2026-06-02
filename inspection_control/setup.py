import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'inspection_control'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'assets'), glob('assets/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your-email@example.com',
    description='A ROS2 package that translates Joy messages to TwistStamped messages at a fixed rate',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'teleop = inspection_control.nodes.teleop_node:main',
            'focus_monitor = inspection_control.nodes.focus_monitor_node:main',
            'multi_speed_autofocus = inspection_control.nodes.multi_speed_autofocus_node:main',
            'autofocus_node = inspection_control.nodes.autofocus_node:main',
            'admittance_control_node = inspection_control.nodes.admittance_control_node:main',
            'orientation_control_node = inspection_control.nodes.orientation_control_node:main',
            'turntable_joy_node = inspection_control.nodes.turntable_joy_node:main',
            'tsdf_pose_node = inspection_control.nodes.tsdf_pose_node:main',
            'servo_logger = inspection_control.nodes.servo_logger_node:main',
        ],
    },
)
