# Numpy version (CPU)
import numpy as np
from scipy.spatial import KDTree
import os



# Point cloud data
point_cloud = np.random.rand(10000, 3) * 3  # Random points in 3x3 meter space

# Grid centers (3x3 grid)
grid_centers = np.array([
    [0.5, 0.5, 0], [1.5, 0.5, 0], [2.5, 0.5, 0],
    [0.5, 1.5, 0], [1.5, 1.5, 0], [2.5, 1.5, 0],
    [0.5, 2.5, 0], [1.5, 2.5, 0], [2.5, 2.5, 0]
])

# Build KDTree
tree = KDTree(point_cloud[:, :2])  # Using only X and Y coordinates

# For each grid center, find points within 50mm and compute average Z
grid_z_values = []
for center in grid_centers:
    indices = tree.query_ball_point(center[:2], r=0.05)  # 50mm radius
    if indices:
        avg_z = np.mean(point_cloud[indices, 2])
        grid_z_values.append(avg_z)
    else:
        grid_z_values.append(0)  # No points found

# Calculate slopes between adjacent points
# ...

import torch

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")

# Get device count
if torch.cuda.is_available():
    print(f"Number of devices: {torch.cuda.device_count()}")
    
    # Get current device
    print(f"Current device: {torch.cuda.current_device()}")
    
    # Get device name
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Try a simple tensor operation
    x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    print(f"Tensor on device: {x.device}")

# Move point cloud to GPU
point_cloud = torch.rand(10000, 3, device="cuda") * 3

# Grid centers
grid_centers = torch.tensor([
    [0.5, 0.5, 0], [1.5, 0.5, 0], [2.5, 0.5, 0],
    [0.5, 1.5, 0], [1.5, 1.5, 0], [2.5, 1.5, 0],
    [0.5, 2.5, 0], [1.5, 2.5, 0], [2.5, 2.5, 0]
], device="cuda")

# For each grid center, compute distances and find points within 50mm
grid_z_values = []
for center in grid_centers:
    # Calculate squared distances (more efficient than Euclidean)
    distances = torch.sum((point_cloud[:, :2] - center[:2])**2, dim=1)
    mask = distances <= 0.05**2  # Points within 50mm
    
    if torch.any(mask):
        # Average Z of points within radius
        avg_z = torch.mean(point_cloud[mask, 2])
        grid_z_values.append(avg_z.item())
    else:
        grid_z_values.append(0)

# Convert to tensor if needed for further GPU calculations
grid_z_values = torch.tensor(grid_z_values, device="cuda")

# Calculate slopes between adjacent points
# ...

# More efficient batch computation for all grid points at once
# Compute pairwise distances between all grid centers and all points
expanded_centers = grid_centers[:, None, :2]  # Shape: [9, 1, 2]
expanded_points = point_cloud[None, :, :2]    # Shape: [1, 10000, 2]
distances = torch.sum((expanded_centers - expanded_points)**2, dim=2)  # [9, 10000]

# Find points within radius for each center
radius_sq = 0.05**2
masks = distances <= radius_sq  # [9, 10000] boolean mask

# Compute average Z for each center
grid_z_values = []
for i, mask in enumerate(masks):
    if torch.any(mask):
        avg_z = torch.mean(point_cloud[mask, 2])
        grid_z_values.append(avg_z.item())
    else:
        grid_z_values.append(0)