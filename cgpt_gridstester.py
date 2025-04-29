import torch
import matplotlib.pyplot as plt
import numpy as np

# Assume gridcomps.py defines these
from gridcomps import point_cloud, grid_centers

# Settings
radius = 0.05  # 50mm radius

# Batch distance calculation (already optimized)
expanded_centers = grid_centers[:, None, :2]  # [9, 1, 2]
expanded_points = point_cloud[None, :, :2]    # [1, N, 2]
distances = torch.sum((expanded_centers - expanded_points)**2, dim=2)  # [9, N]

masks = distances <= radius**2

# Average Z computation
grid_z_values = []
for i, mask in enumerate(masks):
    if torch.any(mask):
        avg_z = torch.mean(point_cloud[mask, 2])
        grid_z_values.append(avg_z.item())
    else:
        grid_z_values.append(0)

grid_z_values = torch.tensor(grid_z_values, device="cuda")  # [9]

# Now reshape into 3x3 for visualization
z_grid = grid_z_values.reshape(3,3).cpu().numpy()

# Calculate slopes between adjacent grid points
slope_x = np.zeros((3,2))  # Between (row,col) and (row,col+1)
slope_y = np.zeros((2,3))  # Between (row,col) and (row+1,col)

# Horizontal slopes (X direction)
for i in range(3):
    for j in range(2):
        dz = z_grid[i,j+1] - z_grid[i,j]
        slope_x[i,j] = dz  # Assume 1 meter spacing in X (can adjust)

# Vertical slopes (Y direction)
for i in range(2):
    for j in range(3):
        dz = z_grid[i+1,j] - z_grid[i,j]
        slope_y[i,j] = dz

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: Point cloud and grid
axs[0].scatter(point_cloud[:,0].cpu(), point_cloud[:,1].cpu(), s=0.5, alpha=0.5)
axs[0].scatter(grid_centers[:,0].cpu(), grid_centers[:,1].cpu(), color='red', marker='x', s=50)
axs[0].set_title('Point Cloud and Grid Centers')
axs[0].set_aspect('equal')

# Panel 2: Heatmap of Z values
im1 = axs[1].imshow(z_grid, origin='lower', extent=[0,3,0,3], cmap='viridis')
axs[1].set_title('Grid Average Z')
fig.colorbar(im1, ax=axs[1])

# Panel 3: Heatmap of slopes
# For simplicity, show average slope magnitude
slope_mag = np.zeros((3,3))
slope_mag[0:3,0:2] += np.abs(slope_x)
slope_mag[0:2,0:3] += np.abs(slope_y)

im2 = axs[2].imshow(slope_mag, origin='lower', extent=[0,3,0,3], cmap='plasma')
axs[2].set_title('Grid Slope Magnitudes')
fig.colorbar(im2, ax=axs[2])

plt.tight_layout()
plt.show()
