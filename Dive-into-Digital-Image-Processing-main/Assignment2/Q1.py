import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load .mat file
# For part a use 'data/points2D_Set1.mat'
file = h5py.File('points2D_Set2.mat', 'r')
# Plot to show how points are scattered
X=file['x']
Y=file['y']
plt.scatter(X,Y)
# Calculate the slope and intercept of the line using linear regression
slope = np.cov(X, Y, ddof=0)[0, 1] / np.var(X, ddof=0)
intercept = np.mean(Y) - slope * np.mean(X)

# Create the line using the calculated slope and intercept
line_x = np.array([np.min(X), np.max(X)])
line_y = slope * line_x + intercept

# Overlay the line on the scatter plot
plt.plot(line_x, line_y, color='red')

# Set labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter plot with linear regression line')

plt.show()