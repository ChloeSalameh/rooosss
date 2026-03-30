from setuptools import setup
import os
from glob import glob

package_name = 'projet'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name], # Modifié pour correspondre à ton dossier
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # CORRECTION : Ajout de '*' pour trouver tous les fichiers .launch.py
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='turtle',
    maintainer_email='turtle@todo.todo',
    description='Package de détection pour le projet 2025',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # CORRECTION : On utilise le nom du package 'projet'
            'line_detector = projet.line_detector:main',
            'superviseur = projet.superviseur:main',
        ],
    },
)