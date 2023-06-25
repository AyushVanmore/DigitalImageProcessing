import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load .mat file
# For part a use 'data/points2D_Set1.mat'
file = h5py.File('data/points2D_Set1.mat', 'r')

# Perform linear regression
try:
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
except ValueError:
    print("Error: Unable to perform linear regression. Please check the input data.")
    exit()


# Plot to show how points are scattered
X=file['x']
Y=file['y']
plt.scatter(X,Y)
# Plot the regression line
plt.plot(x, intercept + slope * x, color='red', label='Linear Regression')

plt.show()