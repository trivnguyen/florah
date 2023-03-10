
import shutil
from setuptools import setup

shutil.copy("config.ini", "src/florah/config.ini")
setup()
