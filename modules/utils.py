import sys
import os
PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PATH_TO_PROJECT)
import pickle as pkl
import numpy as np

def load_pickle(path):
    infile = open(path, 'rb')
    dataset_partitions = pkl.load(infile)
    return dataset_partitions

if __name__ == '__main__':
  a = load_pickle('/home/ereyes/Projects/Harvard/HCGAN/TSTR_data/datasets_original/REAL/9classes_100_100/catalina_north9classes.pkl')
  b = a[0]
  print(b.keys())
  c = b['time_random']
  print(np.array(c).shape)