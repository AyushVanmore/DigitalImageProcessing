import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load .mat file
file = h5py.File('mnist.mat', 'r')

labels=np.array(file['labels_train']).T
images=np.array(file['digits_train']).T

# Reshape the images array to (num_examples, 28, 28)
images_reshaped = np.transpose(images, (2, 0, 1))

# Compute mean for each digit
digit_means = []
for digit in range(10):
    digit_images = images_reshaped[labels_reshaped[:, 0] == digit]
    digit_mean = np.mean(digit_images, axis=0)
    digit_means.append(digit_mean)

# Compute covariance matrix for each digit
digit_covariances = []
for digit in range(10):
    digit_images = images_reshaped[labels_reshaped[:, 0] == digit]
    digit_flattened = digit_images.reshape(digit_images.shape[0], -1).astype(float)
    digit_covariance = np.cov(digit_flattened.T)
    digit_covariances.append(digit_covariance)

# Compute principal mode of variation for each digit
digit_eigenvalues = []
digit_eigenvectors = []
for digit in range(10):
    digit_covariance = digit_covariances[digit]
    eigenvalues, eigenvectors = np.linalg.eig(digit_covariance)
    digit_eigenvalues.append(eigenvalues)
    digit_eigenvectors.append(eigenvectors)

# Plot eigenvalues for each digit
for digit in range(10):
    eigenvalues = digit_eigenvalues[digit]
    plt.plot(range(len(eigenvalues)), eigenvalues, label=f"Digit {digit}")
plt.xlabel('Eigenvalue Index')
plt.ylabel('Eigenvalue')
plt.legend()
plt.show()




# Accessing 50th label from the dataset
print(labels[50])

# Showing 50th image from the dataset
Image.fromarray(images[:,:,50]).show()
