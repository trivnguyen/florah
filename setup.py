
import shutil
from setuptools import setup

shutil.copy("config.ini", "src/merger_tree_ml/config.ini")
setup()
