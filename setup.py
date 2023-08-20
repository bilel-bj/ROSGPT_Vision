from setuptools import setup
from glob import glob
import os

package_name = 'rosgpt_vision'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'webapp'), glob('webapp/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Bilel Benjdira',
    author_email='bilel.benjdira@gmail.com',
    maintainer='Anas M. Ali',
    maintainer_email='anasmagdyhxh@gmail.com',
    keywords=['ROS', 'Vision Language Models'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Creative Commons Attribution-NonCommercial License (CC BY-NC)',
        'Programming Language :: Python :: 3.10', #could work with other version. Tested with 3.10
    ],
    description='A ROS2 package to interact with images Natural Language ',
    license='Creative Commons Attribution-NonCommercial (CC BY-NC)',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ROSGPT_Vision_Camera_Node = rosgpt_vision.ROSGPT_Vision_Camera_Node:main',
            'ROSGPT_Vision_GPT_Consultation_Node = rosgpt_vision.ROSGPT_Vision_GPT_Consultation_Node:main',
        ],
    },
)
