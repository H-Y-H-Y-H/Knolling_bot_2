import torch

# Create bool tensor A with the shape of (8, 30, 1)
A = torch.randint(0, 2, (8, 30, 1), dtype=torch.bool)

# Create bool tensor B with the shape of (8, 1, 30)
B = A.transpose(1, 2)

# Expand the dimensions of A and B
A_expanded = A.expand(-1, -1, 30)
B_expanded = B.expand(-1, 30, -1)

# Calculate the outer product using logical AND
result = A_expanded & B_expanded

# Print the result
print(result)


import torch

# Create a sample (8, 30, 30) tensor
tensor = torch.randn(8, 30, 30)  # Replace with your actual tensor

# Create a mask for elements equal to 0.5
mask = tensor >= 0.5

# Use the mask to select elements not equal to 0.5
filtered_tensor = tensor[mask]

# Reshape the filtered tensor back to (8, 30, 30)
filtered_tensor = filtered_tensor.view(8, 30, 30)

# Print the filtered tensor
print(filtered_tensor)
