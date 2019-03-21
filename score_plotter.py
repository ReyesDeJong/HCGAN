import os

import numpy as np
import pickle

import matplotlib.pyplot as plt

tstr_path = os.path.join(os.getcwd(), 'tstr.pkl')
trts_path = os.path.join(os.getcwd(), 'trts.pkl')
trts_augmented_path = os.path.join(os.getcwd(), 'trts_augmented_90k.pkl')

with open(tstr_path, "rb") as f:
    tstr_dict = pickle.load(f)

with open(trts_path, "rb") as f:
    trts_dict = pickle.load(f)

with open(trts_augmented_path, "rb") as f:
    trts_augmented_dict = pickle.load(f)

n_classes = np.sort(list(tstr_dict.keys()))
tstr_array = []
trts_array = []
trts_augmented_array = []
n_class_array = []

for n_class in n_classes:
    n_int_class = int(n_class)
    n_class_array.append(n_int_class)
    tstr_array.append(tstr_dict[n_class]['testing']['test_onreal_acc'])
    trts_array.append(trts_dict[n_class]['testing']['test_onreal_acc'])
    trts_augmented_array.append(trts_augmented_dict[n_class]['testing']['test_onreal_acc'])

plt.figure()
plt.plot(n_class_array, tstr_array, 'bo', label='TSTR accuracy')
plt.plot(n_class_array, trts_array, 'ro', label ='TRTR Accuracy')
plt.plot(n_class_array, trts_augmented_array, 'go', label='Augmented')
plt.xlabel('Number of classes')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Classifier performance on fake & real datasets')
# plt.grid()
plt.show()
