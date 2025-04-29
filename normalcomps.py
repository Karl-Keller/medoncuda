# NumPy version (CPU)
import numpy as np
from scipy.spatial import KDTree

# Compute normals for each point in the point cloud
def compute_normals_numpy(point_cloud, k=10):
    tree = KDTree(point_cloud)
    normals = np.zeros_like(point_cloud)
    
    for i, point in enumerate(point_cloud):
        # Find k nearest neighbors
        distances, indices = tree.query(point, k=k+1)  # +1 because the point itself is included
        neighbors = point_cloud[indices[1:]]  # Exclude the point itself
        
        # Center the neighbors
        centered = neighbors - point
        
        # Compute covariance matrix
        cov = np.cov(centered, rowvar=False)
        
        # Compute eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # The normal is the eigenvector corresponding to the smallest eigenvalue
        normal = eigenvectors[:, 0]
        
        # Normalize
        normal = normal / np.linalg.norm(normal)
        normals[i] = normal
    
    return normals

# Compute aggregate normals for grid cells
def compute_grid_normals_numpy(point_cloud, normals, grid_centers, radius=0.05):
    # Create a KD-tree with just the x,y coordinates of the point cloud
    tree = KDTree(point_cloud[:, :2])  # Use only x,y coordinates for the tree
    grid_normals = np.zeros((len(grid_centers), 3))
    
    for i, center in enumerate(grid_centers):
        # Query using just the x,y coordinates
        indices = tree.query_ball_point(center[:2], r=radius)
        if indices:
            # Average the normals
            grid_normals[i] = np.mean(normals[indices], axis=0)
            # Normalize
            norm = np.linalg.norm(grid_normals[i])
            if norm > 0:  # Avoid division by zero
                grid_normals[i] = grid_normals[i] / norm
    
    return grid_normals

# Compute normals for each point in the point cloud
def compute_normals_torch(point_cloud, k=10):
    # Move to GPU
    point_cloud_gpu = torch.tensor(point_cloud, device="cuda")
    n_points = point_cloud_gpu.shape[0]
    normals = torch.zeros_like(point_cloud_gpu)
    
    # Compute pairwise distances
    expanded_points1 = point_cloud_gpu.unsqueeze(1).repeat(1, n_points, 1)  # [n, n, 3]
    expanded_points2 = point_cloud_gpu.unsqueeze(0).repeat(n_points, 1, 1)  # [n, n, 3]
    squared_dists = torch.sum((expanded_points1 - expanded_points2)**2, dim=2)  # [n, n]
    
    # Get k nearest neighbors for each point
    _, indices = torch.topk(squared_dists, k=k+1, dim=1, largest=False)
    
    for i in range(n_points):
        # Get neighbors (excluding the point itself)
        neighbor_indices = indices[i, 1:k+1]
        neighbors = point_cloud_gpu[neighbor_indices]
        
        # Center the neighbors
        centered = neighbors - point_cloud_gpu[i]
        
        # Compute covariance matrix
        cov = torch.matmul(centered.t(), centered) / (k - 1)
        
        # Compute eigenvectors and eigenvalues
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        
        # The normal is the eigenvector corresponding to the smallest eigenvalue
        normal = eigenvectors[:, 0]
        
        # Normalize
        normal = normal / torch.norm(normal)
        normals[i] = normal
    
    return normals

# Compute aggregate normals for grid cells
def compute_grid_normals_torch(point_cloud, normals, grid_centers, radius=0.05):
    # Move to GPU
    point_cloud_gpu = torch.tensor(point_cloud, dtype=torch.float32, device="cuda")
    normals_gpu = torch.tensor(normals, dtype=torch.float32, device="cuda")
    grid_centers_gpu = torch.tensor(grid_centers, dtype=torch.float32, device="cuda")
    
    grid_normals = torch.zeros((len(grid_centers_gpu), 3), device="cuda")
    
    # For each grid center
    for i, center in enumerate(grid_centers_gpu):
        # Calculate distances to all points (using only x,y)
        dists = torch.sum((point_cloud_gpu[:, :2] - center[:2])**2, dim=1)
        mask = dists <= radius**2
        
        if torch.any(mask):
            # Average the normals within radius
            grid_normals[i] = torch.mean(normals_gpu[mask], dim=0)
            # Normalize
            norm = torch.norm(grid_normals[i])
            if norm > 0:  # Avoid division by zero
                grid_normals[i] = grid_normals[i] / norm
    
    return grid_normals

import torch
import faiss
import faiss.contrib.torch_utils  # Enable PyTorch integration

# More efficient normal computation using faiss for KNN
def compute_normals_torch_optimized(point_cloud, k=10):
    # Convert to float32 and move to GPU
    point_cloud_gpu = torch.tensor(point_cloud, dtype=torch.float32, device="cuda")
    n_points = point_cloud_gpu.shape[0]
    
    # Create GPU index
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatL2(res, 3)  # 3D points
    
    # Explicitly ensure the data is in float32 format
    index.add(point_cloud_gpu)
    
    # Query for k nearest neighbors
    _, indices = index.search(point_cloud_gpu, k+1)  # [n_points, k+1]
    
    normals = torch.zeros_like(point_cloud_gpu)
    
    # Process in batches for better GPU utilization
    batch_size = 1024
    for batch_start in range(0, n_points, batch_size):
        batch_end = min(batch_start + batch_size, n_points)
        batch_indices = indices[batch_start:batch_end, 1:k+1]  # Exclude the point itself
        
        # Get batched neighbors
        batch_neighbors = point_cloud_gpu[batch_indices]  # [batch_size, k, 3]
        batch_points = point_cloud_gpu[batch_start:batch_end].unsqueeze(1)  # [batch_size, 1, 3]
        
        # Center the neighbors
        centered = batch_neighbors - batch_points
        
        # Process each point in the batch
        for i in range(batch_end - batch_start):
            # Compute covariance matrix for this point
            cov = torch.matmul(centered[i].t(), centered[i]) / (k - 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            normal = eigenvectors[:, 0]
            normal = normal / torch.norm(normal)
            normals[batch_start + i] = normal
    
    return normals

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