import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from normalcomps import (
    compute_normals_numpy, 
    compute_grid_normals_numpy,
    compute_normals_torch, 
    compute_grid_normals_torch,
    compute_normals_torch_optimized
)

# Generate a synthetic point cloud (a hemisphere)
def generate_hemisphere(n_points=1000, radius=1.0, noise=0.05):
    # Generate random points on a hemisphere
    phi = np.random.uniform(0, np.pi/2, n_points)  # Elevation angle
    theta = np.random.uniform(0, 2*np.pi, n_points)  # Azimuth angle
    
    # Convert spherical to Cartesian coordinates
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    
    # Add some noise
    x += np.random.normal(0, noise, n_points)
    y += np.random.normal(0, noise, n_points)
    z += np.random.normal(0, noise, n_points)
    
    point_cloud = np.column_stack((x, y, z))
    
    # The true normals would point outward from the center
    true_normals = point_cloud / np.linalg.norm(point_cloud, axis=1)[:, np.newaxis]
    
    return point_cloud, true_normals

# Generate grid centers for aggregated normals
def generate_grid_centers(x_range, y_range, grid_size):
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    xx, yy = np.meshgrid(x, y)
    
    # Project grid points onto the hemisphere surface (approximate z)
    grid_centers = np.zeros((grid_size*grid_size, 3))
    grid_centers[:, 0] = xx.flatten()
    grid_centers[:, 1] = yy.flatten()
    
    # Estimate z coordinate on hemisphere surface
    r_squared = xx**2 + yy**2
    valid_mask = r_squared <= 1.0
    zz = np.zeros_like(xx)
    zz[valid_mask] = np.sqrt(1.0 - r_squared[valid_mask])
    grid_centers[:, 2] = zz.flatten()
    
    return grid_centers, valid_mask

# Benchmark and compare methods
def benchmark_normal_computation(point_cloud, k=20):
    methods = {
        "NumPy": compute_normals_numpy,
        "PyTorch": compute_normals_torch,
        "PyTorch+FAISS": compute_normals_torch_optimized
    }
    
    results = {}
    
    for name, method in methods.items():
        print(f"Computing normals using {name}...")
        start_time = time.time()
        normals = method(point_cloud, k=k)
        
        # Convert PyTorch tensors to NumPy if needed
        if hasattr(normals, 'cpu'):
            normals = normals.cpu().numpy()
            
        elapsed = time.time() - start_time
        print(f"{name} took {elapsed:.4f} seconds")
        
        results[name] = {
            "normals": normals,
            "time": elapsed
        }
    
    return results

# Plot the point cloud with normals
def plot_results(point_cloud, normals, grid_centers=None, grid_normals=None, subsample=5):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Subsample for visualization
    idx = np.arange(0, len(point_cloud), subsample)
    
    # Plot the point cloud
    ax.scatter(point_cloud[idx, 0], point_cloud[idx, 1], point_cloud[idx, 2], c='b', s=10, alpha=0.5)
    
    # Plot normals as lines
    for i in idx:
        ax.quiver(
            point_cloud[i, 0], point_cloud[i, 1], point_cloud[i, 2],
            normals[i, 0], normals[i, 1], normals[i, 2],
            color='r', length=0.2
        )
    
    # Plot grid normals if provided
    if grid_centers is not None and grid_normals is not None:
        # Subsample grid centers for visualization
        grid_idx = np.arange(0, len(grid_centers), 4)
        
        ax.scatter(
            grid_centers[grid_idx, 0], 
            grid_centers[grid_idx, 1], 
            grid_centers[grid_idx, 2], 
            c='g', s=30, alpha=0.7
        )
        
        for i in grid_idx:
            if np.any(grid_normals[i]):  # Only plot if normal is not zero
                ax.quiver(
                    grid_centers[i, 0], grid_centers[i, 1], grid_centers[i, 2],
                    grid_normals[i, 0], grid_normals[i, 1], grid_normals[i, 2],
                    color='y', length=0.3
                )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud with Computed Normals')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Parameters
    n_points = 5000
    k_neighbors = 20
    grid_size = 20
    
    # Generate synthetic data
    print(f"Generating hemisphere with {n_points} points...")
    point_cloud, true_normals = generate_hemisphere(n_points)
    
    # Calculate grid centers
    grid_centers, valid_mask = generate_grid_centers([-1, 1], [-1, 1], grid_size)
    
    # Benchmark normal computation methods
    results = benchmark_normal_computation(point_cloud, k=k_neighbors)
    
    # Choose one method's output for further processing
    chosen_method = "PyTorch+FAISS"  # Typically the fastest
    computed_normals = results[chosen_method]["normals"]
    
    # Compute grid normals using the CPU implementation
    print("Computing grid normals...")
    start_time = time.time()
    grid_normals_cpu = compute_grid_normals_numpy(point_cloud, computed_normals, grid_centers, radius=0.2)
    elapsed = time.time() - start_time
    print(f"Grid normal computation took {elapsed:.4f} seconds")
    
    # Optional: Compute grid normals using the GPU implementation
    start_time = time.time()
    grid_normals_gpu = compute_grid_normals_torch(point_cloud, computed_normals, grid_centers, radius=0.2)
    if hasattr(grid_normals_gpu, 'cpu'):
        grid_normals_gpu = grid_normals_gpu.cpu().numpy()
    elapsed = time.time() - start_time
    print(f"GPU grid normal computation took {elapsed:.4f} seconds")
    
    # Plot the results
    print("Plotting results...")
    plot_results(point_cloud, computed_normals, grid_centers, grid_normals_gpu)
    
    # Calculate error against true normals
    # Note: We need to handle the sign ambiguity in normal direction
    dot_products = np.abs(np.sum(computed_normals * true_normals, axis=1))
    mean_alignment = np.mean(dot_products)
    print(f"Mean normal alignment with true normals: {mean_alignment:.4f} (1.0 is perfect)")