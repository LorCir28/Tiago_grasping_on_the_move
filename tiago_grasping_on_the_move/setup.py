from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
d = generate_distutils_setup(
    packages=['tiago_grasping_on_the_move'],
    package_dir={'': 'src'}
)
setup(**d)