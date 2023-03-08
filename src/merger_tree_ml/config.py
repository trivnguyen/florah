
import configparser
import os

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), "config.ini"))

DEFAULT_RUN_PATH = config.get("ENVIRONMENT_VARIABLES", "DEFAULT_RUN_PATH")
DEFAULT_DATASET_PATH = config.get(
    "ENVIRONMENT_VARIABLES", "DEFAULT_DATASET_PATH")
DEFAULT_RAW_DATASET_PATH = config.get(
    "ENVIRONMENT_VARIABLES", "DEFAULT_RAW_DATASET_PATH")
BP_TABLE_PATH = '/mnt/home/tnguyen/merger_tree_ml/tables/bp_redshifts.txt'
GUREFT_TABLE_PATH = '/mnt/home/tnguyen/merger_tree_ml/tables/gureft_redshifts.txt'
VSMDPL_TABLE_PATH = '/mnt/home/tnguyen/merger_tree_ml/tables/vsmdpl_redshifts.txt'
