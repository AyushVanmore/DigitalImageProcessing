import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load .mat file
file = h5py.File('data/mnist.mat', 'r')

labels=np.array(file['labels_train']).T
images=np.array(file['digits_train']).T

# Accessing 50th label from the dataset
print(labels[50])

# Showing 50th image from the dataset
Image.fromarray(images[:,:,50]).show()