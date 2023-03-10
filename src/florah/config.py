
import configparser
import os

# read in environmental variables from config file
config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), "config.ini"))
DEFAULT_RUN_PATH = config.get("ENVIRONMENT_VARIABLES", "DEFAULT_RUN_PATH")
DEFAULT_DATASET_PATH = config.get(
    "ENVIRONMENT_VARIABLES", "DEFAULT_DATASET_PATH")
DEFAULT_RAW_DATASET_PATH = config.get(
    "ENVIRONMENT_VARIABLES", "DEFAULT_RAW_DATASET_PATH")

# define more environmental variables
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")