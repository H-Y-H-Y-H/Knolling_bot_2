import numpy as np
import matplotlib.pyplot as plt

# Define the size of the grid
grid_size = 5
x_range = (-0.01, 0.01)
y_range = (-0.01, 0.01)

# Create a grid of x and y values
x_values = np.linspace(x_range[0], x_range[1], grid_size)
y_values = np.linspace(y_range[0], y_range[1], grid_size)
xx, yy = np.meshgrid(x_values, y_values)

# Define the position of the point you want to blur
point_x = 0.
point_y = 0.

# Define the standard deviation for the Gaussian distribution (controls the amount of blur)
sigma = 0.01

# Calculate the Gaussian kernel centered at the point
kernel = np.exp(-((xx - point_x)**2 + (yy - point_y)**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)

# Normalize the kernel
kernel = kernel / np.sum(kernel)

# Plot the Gaussian kernel
plt.imshow(kernel, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], origin='lower', cmap='viridis')
plt.colorbar()
plt.title('Gaussian Kernel')
plt.show()

import numpy as np

# Create your 10x10 value matrix and probability matrix (replace with your data)
value_matrix = np.arange(25).reshape(5, 5) + 500
probability_matrix = kernel

# Flatten the matrices
flattened_value = value_matrix.flatten()
flattened_prob = probability_matrix.flatten()

# Sample 5 indices based on the probabilities
selected_indices = np.random.choice(len(flattened_prob), 5, p=flattened_prob)

# Map the selected indices back to 2D coordinates
selected_points = [(index // 10, index % 10) for index in selected_indices]

# Ensure the selected points are different
selected_points = list(set(selected_points))

# Print the selected points
print("Selected Points:")
for point in selected_points:
    print(f"Value: {value_matrix[point]}, Probability: {probability_matrix[point]}")
