import pickle as pkl
import numpy as np
import pandas as pd
import os
import sys
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, \
    load_robot_execution_failures
PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PATH_TO_PROJECT)

if __name__ == '__main__':
    download_robot_execution_failures()
    timeseries, y = load_robot_execution_failures()