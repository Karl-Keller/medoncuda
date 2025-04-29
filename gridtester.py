import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time
import torch
from scipy.spatial import KDTree

# Set OpenMP environment variable to avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def generate_terrain_point_cloud(n_points=50000, 
                                 x_range=(0, 3), 
                                 y_range=(0, 3), 
                                 noise_level=0.05,
                                 seed=42):
    """Generate a synthetic terrain-like point cloud with ridges and valleys."""
    np.random.seed(seed)
    
    # Generate random x, y coordinates
    x = np.random.uniform(x_range[0], x_range[1], n_points)
    y = np.random.uniform(y_range[0], y_range[1], n_points)
    
    # Create a more professional-looking terrain with ridges and valleys
    z_base = 0.2 * np.sin(x * 3.5) * np.cos(y * 2.5) + 0.15 * np.cos((x - y) * 4)
    
    # Add some random hills - multiple smaller ones instead of central mounds
    hill_centers = [
        (0.5, 0.5), (2.5, 0.8), (1.7, 2.2), (0.8, 2.7)
    ]
    
    z_hills = np.zeros_like(x)
    for cx, cy in hill_centers:
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        # Smaller, more distributed hills
        z_hills += 0.2 * np.exp(-dist**2 / 0.3)
    
    # Add a ridge line
    ridge_dist = np.abs(y - 0.5 * x - 1.0)
    z_ridge = 0.3 * np.exp(-ridge_dist**2 / 0.05)
    
    # Add random noise
    z_noise = np.random.normal(0, noise_level, n_points)
    
    # Combine components
    z = z_base + z_hills + z_ridge + z_noise
    
    # Create point cloud
    point_cloud = np.column_stack((x, y, z))
    
    return point_cloud

def create_grid_from_extremes(point_cloud, step_size=0.05):
    """Create a regular grid based on point cloud extremes with specified step size."""
    # Find extremes of point cloud
    x_min, y_min = np.min(point_cloud[:, :2], axis=0)
    x_max, y_max = np.max(point_cloud[:, :2], axis=0)
    
    # Create grid based on extremes
    x = np.arange(x_min, x_max + step_size, step_size)
    y = np.arange(y_min, y_max + step_size, step_size)
    
    grid_size_x = len(x)
    grid_size_y = len(y)
    
    xx, yy = np.meshgrid(x, y)
    
    # Initialize z-values to zero
    zz = np.zeros_like(xx)
    
    # Create grid centers as 3D points [x, y, z]
    grid_centers = np.zeros((grid_size_y * grid_size_x, 3))
    grid_centers[:, 0] = xx.flatten()
    grid_centers[:, 1] = yy.flatten()
    
    grid_info = {
        'centers': grid_centers,
        'xx': xx,
        'yy': yy,
        'zz': zz,
        'size_x': grid_size_x,
        'size_y': grid_size_y,
        'step_size': step_size
    }
    
    return grid_info

def compute_grid_z_numpy(point_cloud, grid_info, radius=0.05):
    """Compute average z-values for grid points using NumPy and KDTree."""
    # Build KDTree using only X and Y coordinates
    tree = KDTree(point_cloud[:, :2])
    
    grid_centers = grid_info['centers']
    
    # For each grid center, find points within radius and compute average Z
    for i, center in enumerate(grid_centers):
        indices = tree.query_ball_point(center[:2], r=radius)
        if indices:
            grid_centers[i, 2] = np.mean(point_cloud[indices, 2])
    
    # Reshape z-values to grid
    zz = grid_centers[:, 2].reshape(grid_info['size_y'], grid_info['size_x'])
    
    return zz

def compute_grid_z_torch(point_cloud, grid_info, radius=0.1):
    """Compute average z-values for grid points using PyTorch with batch processing."""
    # Move data to GPU
    point_cloud_gpu = torch.tensor(point_cloud, dtype=torch.float32, device="cuda")
    grid_centers = grid_info['centers']
    grid_centers_gpu = torch.tensor(grid_centers, dtype=torch.float32, device="cuda")
    
    # Process in batches for better performance
    batch_size = 100  # Process this many grid points at once
    
    for batch_start in range(0, len(grid_centers_gpu), batch_size):
        batch_end = min(batch_start + batch_size, len(grid_centers_gpu))
        batch_centers = grid_centers_gpu[batch_start:batch_end]
        
        # Compute pairwise distances between batch centers and all points
        expanded_centers = batch_centers[:, None, :2]  # [batch, 1, 2]
        expanded_points = point_cloud_gpu[None, :, :2]  # [1, n_points, 2]
        
        # Compute squared distances
        distances = torch.sum((expanded_centers - expanded_points)**2, dim=2)  # [batch, n_points]
        
        # Find points within radius for each center
        masks = distances <= radius**2  # [batch, n_points]
        
        # Compute average Z for each center in the batch
        for i in range(len(batch_centers)):
            mask = masks[i]
            if torch.any(mask):
                batch_centers[i, 2] = torch.mean(point_cloud_gpu[mask, 2])
        
        # Update the original grid_centers_gpu with computed z-values
        grid_centers_gpu[batch_start:batch_end, 2] = batch_centers[:, 2]
    
    # Copy back to CPU and update grid_centers
    grid_centers[:, 2] = grid_centers_gpu[:, 2].cpu().numpy()
    
    # Reshape z-values to grid
    zz = grid_centers[:, 2].reshape(grid_info['size_y'], grid_info['size_x'])
    
    return zz

def compute_grid_z_torch_optimized(point_cloud, grid_info, radius=0.05):
    """Memory-efficient GPU implementation with chunked processing."""
    # Move data to GPU
    point_cloud_gpu = torch.tensor(point_cloud, dtype=torch.float32, device="cuda")
    point_xy = point_cloud_gpu[:, :2]
    point_z = point_cloud_gpu[:, 2]
    
    grid_centers = grid_info['centers']
    grid_centers_gpu = torch.tensor(grid_centers, dtype=torch.float32, device="cuda")
    
    # Initialize results
    grid_z = torch.zeros(len(grid_centers_gpu), device="cuda")
    
    # Process grid points in small chunks
    grid_chunk_size = 64
    
    # Process point cloud in chunks too
    point_chunk_size = 10000
    
    for grid_start in range(0, len(grid_centers_gpu), grid_chunk_size):
        grid_end = min(grid_start + grid_chunk_size, len(grid_centers_gpu))
        grid_chunk = grid_centers_gpu[grid_start:grid_end, :2]
        
        # Initialize accumulators for this grid chunk
        sum_z = torch.zeros(grid_end - grid_start, device="cuda")
        count = torch.zeros(grid_end - grid_start, device="cuda")
        
        # Process point cloud in chunks
        for point_start in range(0, len(point_cloud_gpu), point_chunk_size):
            point_end = min(point_start + point_chunk_size, len(point_cloud_gpu))
            
            # Extract chunk
            point_xy_chunk = point_xy[point_start:point_end]
            point_z_chunk = point_z[point_start:point_end]
            
            # Calculate pairwise distances (grid_chunk x point_chunk)
            # Reshape for broadcasting
            grid_expanded = grid_chunk.unsqueeze(1)  # [grid_chunk, 1, 2]
            point_expanded = point_xy_chunk.unsqueeze(0)  # [1, point_chunk, 2]
            
            # Compute squared distances
            dist_sq = torch.sum((grid_expanded - point_expanded)**2, dim=2)  # [grid_chunk, point_chunk]
            
            # Get mask of points within radius
            mask = dist_sq <= radius**2  # [grid_chunk, point_chunk]
            
            # Update sums and counts
            # for i in range(len(grid_chunk)):
            #     valid_points = point_z_chunk[mask[i]]
            #     if len(valid_points) > 0:
            #         sum_z[i] += torch.sum(valid_points)
            #         count[i] += len(valid_points)
            # mask: (grid_chunk_size, point_chunk_size)
            # point_z_chunk: (point_chunk_size)

            # Expand Z values to (1, point_chunk_size)
            point_z_expanded = point_z_chunk.unsqueeze(0)  # [1, point_chunk_size]

            # Masked sums across points for each grid center
            sum_z_chunk = torch.sum(point_z_expanded * mask.float(), dim=1)  # [grid_chunk_size]

            # Masked counts
            count_chunk = torch.sum(mask, dim=1)  # [grid_chunk_size]

            # Update totals
            sum_z += sum_z_chunk
            count += count_chunk

        
        # Compute averages for this grid chunk
        valid_mask = count > 0
        grid_z[grid_start:grid_end][valid_mask] = sum_z[valid_mask] / count[valid_mask]
    
    # Reshape to grid
    zz = grid_z.cpu().numpy().reshape(grid_info['size_y'], grid_info['size_x'])
    
    return zz

def compute_slopes_torch_optimized(grid_info):
    """Compute slopes using efficient PyTorch operations."""
    xx = grid_info['xx']
    yy = grid_info['yy']
    zz = grid_info['zz']
    step_size = grid_info['step_size']
    
    # Convert to tensor and move to GPU
    zz_gpu = torch.tensor(zz, dtype=torch.float32, device="cuda")
    
    # Use 2D convolution to compute gradients efficiently
    # Create padding to handle edge cases
    zz_padded = torch.nn.functional.pad(zz_gpu.unsqueeze(0).unsqueeze(0), 
                                        (1, 1, 1, 1), mode='replicate')
    
    # Sobel-like kernels for gradient computation
    kernel_x = torch.tensor([[[[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]]]], dtype=torch.float32, device="cuda")
    
    kernel_y = torch.tensor([[[[-1, -2, -1],
                              [0, 0, 0],
                              [1, 2, 1]]]], dtype=torch.float32, device="cuda")
    
    # Compute gradients using convolution
    grad_x = torch.nn.functional.conv2d(zz_padded, kernel_x) / (8.0 * step_size)
    grad_y = torch.nn.functional.conv2d(zz_padded, kernel_y) / (8.0 * step_size)
    
    # Extract from tensor
    slope_x = grad_x[0, 0].cpu().numpy()
    slope_y = grad_y[0, 0].cpu().numpy()
    
    return slope_x, slope_y

def compute_slopes_advanced(grid_info):
    """
    Compute slopes using specified approach:
    - Edge points: simple slope to neighbors
    - Interior points: ray projection through min/max neighbors
    """
    xx = grid_info['xx']
    yy = grid_info['yy']
    zz = grid_info['zz']
    step_size = grid_info['step_size']
    size_y, size_x = zz.shape
    
    # Initialize slope grids
    slope_x = np.zeros_like(zz)
    slope_y = np.zeros_like(zz)
    
    # Process each grid point
    for i in range(size_y):
        for j in range(size_x):
            # Check if this is an edge point
            is_edge_x = (j == 0 or j == size_x - 1)
            is_edge_y = (i == 0 or i == size_y - 1)
            
            # X-direction slope
            if is_edge_x:
                # Edge point: simple slope
                if j == 0:
                    # Left edge
                    slope_x[i, j] = (zz[i, j+1] - zz[i, j]) / step_size
                else:
                    # Right edge
                    slope_x[i, j] = (zz[i, j] - zz[i, j-1]) / step_size
            else:
                # Interior point: ray projection approach
                # Find min and max neighbors in x-direction
                left_z = zz[i, j-1]
                right_z = zz[i, j+1]
                
                if left_z < right_z:
                    min_z, max_z = left_z, right_z
                    min_x, max_x = xx[i, j-1], xx[i, j+1]
                else:
                    min_z, max_z = right_z, left_z
                    min_x, max_x = xx[i, j+1], xx[i, j-1]
                
                # Calculate slope using ray projection
                dz = max_z - min_z
                dx = max_x - min_x
                slope_x[i, j] = dz / dx if dx != 0 else 0
            
            # Y-direction slope
            if is_edge_y:
                # Edge point: simple slope
                if i == 0:
                    # Top edge
                    slope_y[i, j] = (zz[i+1, j] - zz[i, j]) / step_size
                else:
                    # Bottom edge
                    slope_y[i, j] = (zz[i, j] - zz[i-1, j]) / step_size
            else:
                # Interior point: ray projection approach
                # Find min and max neighbors in y-direction
                top_z = zz[i-1, j]
                bottom_z = zz[i+1, j]
                
                if top_z < bottom_z:
                    min_z, max_z = top_z, bottom_z
                    min_y, max_y = yy[i-1, j], yy[i+1, j]
                else:
                    min_z, max_z = bottom_z, top_z
                    min_y, max_y = yy[i+1, j], yy[i-1, j]
                
                # Calculate slope using ray projection
                dz = max_z - min_z
                dy = max_y - min_y
                slope_y[i, j] = dz / dy if dy != 0 else 0
    
    return slope_x, slope_y

def visualize_results(point_cloud, grid_info, slope_x, slope_y, subsample=10):
    """Visualize point cloud, grid, and slopes with clear distinction between point cloud and grid."""
    xx = grid_info['xx']
    yy = grid_info['yy']
    zz = grid_info['zz']
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(18, 15))
    
    # 3D plot of point cloud and grid surface
    ax1 = fig.add_subplot(231, projection='3d')
    
    # Subsample point cloud for visualization
    idx = np.random.choice(len(point_cloud), size=min(len(point_cloud) // subsample, 1000), replace=False)
    
    # Plot point cloud
    ax1.scatter(point_cloud[idx, 0], point_cloud[idx, 1], point_cloud[idx, 2], 
               c='blue', marker='.', alpha=0.3, label='Original Cloud Points')
    
    # Plot grid as a surface
    surf = ax1.plot_surface(xx, yy, zz, alpha=0.7, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    
    # Set labels and title
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Point Cloud with Grid Surface')
    ax1.legend()
    
    # 3D plot of just the grid points
    ax2 = fig.add_subplot(232, projection='3d')
    
    # Show grid points as markers
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            ax2.scatter(xx[i,j], yy[i,j], zz[i,j], c='red', marker='o', s=10)
    
    # Set labels and title
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Grid Points (X,Y,Z)')
    
    # Top-down view showing both point cloud and grid points
    ax3 = fig.add_subplot(233)
    ax3.scatter(point_cloud[idx, 0], point_cloud[idx, 1], c='blue', marker='.', alpha=0.3, label='Cloud Points')
    
    # Plot grid points
    grid_x = xx.flatten()
    grid_y = yy.flatten()
    ax3.scatter(grid_x, grid_y, c='red', marker='o', s=10, label='Grid Points')
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('Top View: Cloud and Grid Points')
    ax3.legend()
    
    # Plot the grid z-values as a heatmap
    ax4 = fig.add_subplot(234)
    im = ax4.imshow(zz, origin='lower', extent=[xx.min(), xx.max(), yy.min(), yy.max()], 
                    cmap=cm.coolwarm, interpolation='bicubic')
    plt.colorbar(im, ax=ax4, label='Z Value')
    ax4.set_title('Grid Z Values')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    
    # Plot X-slope
    ax5 = fig.add_subplot(235)
    im = ax5.imshow(slope_x, origin='lower', extent=[xx.min(), xx.max(), yy.min(), yy.max()], 
                   cmap=cm.coolwarm, interpolation='bicubic')
    plt.colorbar(im, ax=ax5, label='dZ/dX')
    ax5.set_title('X Slope (dZ/dX)')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    
    # Plot Y-slope
    ax6 = fig.add_subplot(236)
    im = ax6.imshow(slope_y, origin='lower', extent=[xx.min(), xx.max(), yy.min(), yy.max()], 
                   cmap=cm.coolwarm, interpolation='bicubic')
    plt.colorbar(im, ax=ax6, label='dZ/dY')
    ax6.set_title('Y Slope (dZ/dY)')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    
    plt.tight_layout()
    plt.show()

def benchmark_grid_computation(point_cloud, grid_info, radius=0.05, repeat=3):
    """Benchmark different methods for computing grid Z values."""
    methods = {
        "NumPy": compute_grid_z_numpy,
        "PyTorch (Basic)": compute_grid_z_torch,
        "PyTorch (Optimized)": compute_grid_z_torch_optimized
    }
    
    results = {}
    
    # Print dataset size info
    print(f"Point cloud size: {len(point_cloud):,} points")
    print(f"Grid size: {grid_info['size_y']} × {grid_info['size_x']} = {len(grid_info['centers']):,} grid points")
    print(f"Search radius: {radius*1000:.1f}mm\n")
    
    # Run each method multiple times and take minimum time (to reduce timing noise)
    for name, method in methods.items():
        print(f"Computing grid Z values using {name}...")
        times = []
        
        # Warmup run (not timed)
        _ = method(point_cloud, grid_info, radius)
        
        for i in range(repeat):
            start_time = time.time()
            grid_info['zz'] = method(point_cloud, grid_info, radius)
            elapsed = time.time() - start_time
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.4f} seconds")
        
        # Get best time
        best_time = min(times)
        print(f"  Best time: {best_time:.4f} seconds")
        
        results[name] = {
            "time": best_time
        }
        
        # Calculate speedup relative to NumPy
        if name != "NumPy" and "NumPy" in results:
            speedup = results["NumPy"]["time"] / best_time
            print(f"  Speedup vs NumPy: {speedup:.2f}x")
        
        print()
    
    # Report best method
    best_method = min(results.items(), key=lambda x: x[1]["time"])[0]
    print(f"Fastest method: {best_method}")
    
    return results

def benchmark_slope_computation(grid_info, repeat=3):
    """Benchmark different methods for computing slopes."""
    methods = {
        "NumPy": compute_slopes_advanced,
        "PyTorch": compute_slopes_torch_optimized
    }
    
    results = {}
    
    print(f"\nBenchmarking slope computation:")
    print(f"Grid size: {grid_info['size_y']} × {grid_info['size_x']}\n")
    
    # Run each method multiple times and take minimum time
    for name, method in methods.items():
        print(f"Computing slopes using {name}...")
        times = []
        
        # Warmup run
        slope_x, slope_y = method(grid_info)
        
        for i in range(repeat):
            start_time = time.time()
            slope_x, slope_y = method(grid_info)
            elapsed = time.time() - start_time
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.4f} seconds")
        
        # Get best time
        best_time = min(times)
        print(f"  Best time: {best_time:.4f} seconds")
        
        results[name] = {
            "time": best_time,
            "slope_x": slope_x,
            "slope_y": slope_y
        }
        
        # Calculate speedup relative to NumPy
        if name != "NumPy" and "NumPy" in results:
            # Avoid division by zero
            if best_time > 0:
                speedup = results["NumPy"]["time"] / best_time
                print(f"  Speedup vs NumPy: {speedup:.2f}x")
            else:
                print(f"  Speedup vs NumPy: ∞ (too fast to measure accurately)")
        
        print()
    
    return results

import faiss


def compute_grid_z_faiss(point_cloud, grid_info, radius=0.025):
    """Use FAISS (GPU-accelerated) for nearest neighbor search using k-NN and CPU-side averaging."""
    # Create FAISS index (on CPU first for safer initialization)
    index = faiss.IndexFlatL2(2)  # 2D index (x,y only)
    index.add(point_cloud[:, :2].astype(np.float32))  # FAISS requires float32

    # Move to GPU if available
    if hasattr(faiss, 'StandardGpuResources'):
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print("  Using GPU-accelerated FAISS")
        except Exception as e:
            print(f"  Could not use GPU FAISS: {e}. Falling back to CPU.")

    # Initialize results
    grid_z = np.zeros(len(grid_info['centers']))
    radius_sq = radius**2

    # Determine maximum number of neighbors to fetch
    point_density = len(point_cloud) / ((np.max(point_cloud[:, 0]) - np.min(point_cloud[:, 0])) * 
                                        (np.max(point_cloud[:, 1]) - np.min(point_cloud[:, 1])))
    avg_neighbors = int(np.ceil(point_density * np.pi * radius**2 * 2))  # Add safety factor of 2
    k = min(max(30, avg_neighbors), 500)  # Limit between 30 and 500
    print(f"  Using k={k} neighbors for FAISS search")

    # Process grid points in batches
    batch_size = 256
    for i in range(0, len(grid_info['centers']), batch_size):
        end = min(i + batch_size, len(grid_info['centers']))
        query_batch = grid_info['centers'][i:end, :2].astype(np.float32)

        # Use k-NN search and filter by distance
        distances, indices = index.search(query_batch, k)

        # Process each grid point
        for j in range(end - i):
            # Get valid neighbors (within radius and not padding)
            valid_mask = (distances[j] <= radius_sq) & (indices[j] >= 0)
            valid_indices = indices[j][valid_mask]

            if len(valid_indices) > 0:
                grid_z[i+j] = np.mean(point_cloud[valid_indices, 2])

    # Reshape to grid
    zz = grid_z.reshape(grid_info['size_y'], grid_info['size_x'])
    return zz



def compute_grid_z_numpy_parallel(point_cloud, grid_info, radius=0.05, n_jobs=-1):
    """Optimized parallel CPU implementation using NumPy and KDTree."""
    from joblib import Parallel, delayed
    import multiprocessing
    
    # Build KDTree using only X and Y coordinates (done once for all workers)
    tree = KDTree(point_cloud[:, :2])
    
    # Determine number of CPU cores to use
    n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
    print(f"  Using {n_jobs} CPU cores")
    
    # Get total grid points
    n_grid_points = len(grid_info['centers'])
    
    # Split work more evenly with larger chunks
    chunk_size = max(100, n_grid_points // n_jobs)
    
    # Helper function to process a chunk of grid points
    def process_chunk(start_idx, end_idx):
        chunk_results = np.zeros(end_idx - start_idx)
        for i in range(end_idx - start_idx):
            idx = start_idx + i
            center = grid_info['centers'][idx]
            indices = tree.query_ball_point(center[:2], r=radius)
            if indices:
                chunk_results[i] = np.mean(point_cloud[indices, 2])
        return chunk_results
    
    # Create chunks
    chunks = [(i, min(i + chunk_size, n_grid_points)) 
              for i in range(0, n_grid_points, chunk_size)]
    
    # Process chunks in parallel
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(process_chunk)(start, end) for start, end in chunks
    )
    
    # Combine results
    grid_z = np.zeros(n_grid_points)
    for i, (start, end) in enumerate(chunks):
        grid_z[start:end] = results[i]
    
    # Reshape to grid
    zz = grid_z.reshape(grid_info['size_y'], grid_info['size_x'])
    
    return zz

def main():
    # Parameters
    n_points = 10000000     # Half million points
    step_size = 3.00      # Grid step size (50mm)
    radius = 0.05        # 25mm radius for finding nearest neighbors
    x_range = (0, 30)      # X range for terrain
    y_range = (0, 30)      # Y range for terrain
    
    # Check CUDA availability and show memory info
    if torch.cuda.is_available():
        print(f"CUDA available: True")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Current memory usage: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    else:
        print("CUDA not available. Using CPU only.")
    
    # Generate synthetic terrain point cloud
    print(f"Generating terrain point cloud with {n_points:,} points...")
    point_cloud = generate_terrain_point_cloud(n_points, x_range, y_range)
    
    # Calculate point cloud density
    area = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0])
    density = n_points / area
    avg_points_per_cell = density * (np.pi * radius**2)
    print(f"Point cloud density: {density:.1f} points/m² (~ {avg_points_per_cell:.1f} points per grid cell)")
    
    # Create grid based on extremes with specified step size
    print(f"Creating grid with {step_size}m step size...")
    grid_info = create_grid_from_extremes(point_cloud, step_size)
    
    # Define all methods to benchmark
    methods = {
        "NumPy": compute_grid_z_numpy,
        "PyTorch (Basic)": compute_grid_z_torch,
        #"PyTorch (Optimized)": compute_grid_z_torch_optimized,
        "NumPy (Parallel)": lambda pc, gi, r: compute_grid_z_numpy_parallel(pc, gi, r, n_jobs=os.cpu_count())
    }
    
    # Add FAISS method if CUDA is available
    if torch.cuda.is_available():
        try:
            import faiss
            methods["FAISS (GPU)"] = compute_grid_z_faiss
            print("FAISS library detected, adding GPU-accelerated method to benchmark.")
        except ImportError:
            print("FAISS library not found. Skipping GPU-accelerated nearest neighbor method.")
    
    # Print benchmark header
    print(f"Point cloud size: {len(point_cloud):,} points")
    print(f"Grid size: {grid_info['size_y']} × {grid_info['size_x']} = {len(grid_info['centers']):,} grid points")
    print(f"Search radius: {radius*1000:.1f}mm\n")
    
    # Run benchmarks
    results = {}
    best_time = float('inf')
    best_method = None
    
    # Run each method with multiple repeats to get accurate timing
    repeats = 3
    
    for name, method in methods.items():
        print(f"Computing grid Z values using {name}...")
        times = []
        
        try:
            # Warmup run (not timed)
            _ = method(point_cloud, grid_info, radius)
            
            # Timed runs
            for i in range(repeats):
                start_time = time.time()
                grid_info['zz'] = method(point_cloud, grid_info, radius)
                elapsed = time.time() - start_time
                times.append(elapsed)
                print(f"  Run {i+1}: {elapsed:.4f} seconds")
            
            # Calculate statistics
            best_run = min(times)
            print(f"  Best time: {best_run:.4f} seconds")
            
            # Compare to baseline NumPy
            if name != "NumPy" and "NumPy" in results:
                speedup = results["NumPy"]["time"] / best_run
                print(f"  Speedup vs NumPy: {speedup:.2f}x")
            
            # Store results
            results[name] = {
                "time": best_run
            }
            
            # Track best method
            if best_run < best_time:
                best_time = best_run
                best_method = name
        
        except Exception as e:
            print(f"  Error running {name}: {e}")
            print(f"  Skipping this method.")
        
        print()
    
    # Report best method if any succeeded
    if best_method:
        print(f"Fastest method: {best_method} ({best_time:.4f} seconds)")
    else:
        print("No methods completed successfully.")
    
    # Compute slopes using advanced method
    print("Computing slopes using advanced method...")
    slope_x, slope_y = compute_slopes_advanced(grid_info)
    
    # Visualize results
    print("Visualizing results...")
    visualize_results(point_cloud, grid_info, slope_x, slope_y)

if __name__ == "__main__":
    main()