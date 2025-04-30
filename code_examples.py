"""
Point Cloud Processing Library - Usage Examples

This file provides examples of how to use the grid and normal computation modules.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import torch

# Set OpenMP environment variable to avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Import grid computation functions
from gridtester import (
    generate_terrain_point_cloud,
    create_grid_from_extremes,
    compute_grid_z_numpy,
    compute_grid_z_numpy_parallel,
    compute_grid_z_torch,
    compute_grid_z_faiss,
    compute_slopes_advanced,
    visualize_results
)

# Import normal computation functions
from normalcomps import (
    compute_normals_numpy,
    compute_normals_torch,
    compute_normals_torch_optimized,
    compute_grid_normals_numpy,
    compute_grid_normals_torch
)

from normalstester import (
    generate_hemisphere,
    generate_grid_centers,
    plot_results
)

# ===== Example 1: Basic Grid Computation =====

def example_grid_computation():
    """
    Demonstrates basic grid computation with NumPy.
    """
    print("\n===== Example 1: Basic Grid Computation =====")
    
    # Parameters
    n_points = 100000     # 100K points in the cloud
    step_size = 0.05      # 50mm grid step size
    radius = 0.025        # 25mm search radius
    x_range = (0, 3)      # X range for terrain (3m x 3m area)
    y_range = (0, 3)      # Y range for terrain
    
    # Generate a synthetic terrain point cloud
    print(f"Generating terrain point cloud with {n_points} points...")
    point_cloud = generate_terrain_point_cloud(n_points, x_range, y_range)
    
    # Create a grid over the terrain
    print(f"Creating grid with {step_size}m step size...")
    grid_info = create_grid_from_extremes(point_cloud, step_size)
    
    # Compute slopes
    print("Computing slopes...")
    slope_x, slope_y = compute_slopes_advanced(grid_info)
    
    # Visualize results
    print("Visualizing results...")
    visualize_results(point_cloud, grid_info, slope_x, slope_y)
    
    print("Example 4 completed!\n")


# ===== Example 5: Integration in a Real-World Pipeline =====

def example_workflow_pipeline():
    """
    Demonstrates a complete workflow pipeline for point cloud processing.
    """
    print("\n===== Example 5: Complete Workflow Pipeline =====")
    
    # Parameters
    n_points = 300000     # 300K points
    step_size = 0.05      # 50mm grid step size
    radius = 0.025        # 25mm search radius
    x_range = (0, 3)      # X range for terrain
    y_range = (0, 3)      # Y range for terrain
    
    # Step 1: Data Generation/Loading
    print("Step 1: Generating synthetic terrain data...")
    point_cloud = generate_terrain_point_cloud(n_points, x_range, y_range)
    
    # Step 2: Grid Creation
    print("Step 2: Creating analysis grid...")
    grid_info = create_grid_from_extremes(point_cloud, step_size)
    
    # Step 3: Choose best method based on data size and available hardware
    print("Step 3: Selecting optimal computation method...")
    
    if n_points > 1000000 and torch.cuda.is_available():
        try:
            import faiss
            print("  Selected FAISS method for large dataset with GPU")
            compute_method = compute_grid_z_faiss
        except ImportError:
            print("  Selected NumPy Parallel method for large dataset (FAISS not available)")
            compute_method = lambda pc, gi, r: compute_grid_z_numpy_parallel(pc, gi, r, n_jobs=-1)
    elif n_points > 500000:
        print("  Selected NumPy Parallel method for medium dataset")
        compute_method = lambda pc, gi, r: compute_grid_z_numpy_parallel(pc, gi, r, n_jobs=-1)
    else:
        print("  Selected NumPy method for small dataset")
        compute_method = compute_grid_z_numpy
    
    # Step 4: Z-value computation
    print("Step 4: Computing grid Z-values...")
    start_time = time.time()
    grid_info['zz'] = compute_method(point_cloud, grid_info, radius)
    z_time = time.time() - start_time
    print(f"  Z-value computation took {z_time:.4f} seconds")
    
    # Step 5: Slope computation
    print("Step 5: Computing terrain slopes...")
    start_time = time.time()
    slope_x, slope_y = compute_slopes_advanced(grid_info)
    slope_time = time.time() - start_time
    print(f"  Slope computation took {slope_time:.4f} seconds")
    
    # Step 6: Analysis - find steep areas
    print("Step 6: Analyzing terrain for steep slopes...")
    slope_magnitude = np.sqrt(slope_x**2 + slope_y**2)
    steep_mask = slope_magnitude > 1.0  # Slope > 45 degrees (approximately)
    steep_percentage = np.sum(steep_mask) / slope_magnitude.size * 100
    print(f"  {steep_percentage:.2f}% of terrain has slopes steeper than 45 degrees")
    
    # Step 7: Visualization
    print("Step 7: Visualizing results...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 8))
    
    # Plot Z values
    ax1 = fig.add_subplot(121)
    im1 = ax1.imshow(grid_info['zz'], origin='lower', cmap='terrain')
    plt.colorbar(im1, ax=ax1, label='Elevation (m)')
    ax1.set_title('Terrain Elevation')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # Plot slope magnitude with steep areas highlighted
    ax2 = fig.add_subplot(122)
    im2 = ax2.imshow(slope_magnitude, origin='lower', cmap='viridis')
    plt.colorbar(im2, ax=ax2, label='Slope Magnitude')
    
    # Highlight steep areas
    steep_overlay = np.zeros_like(slope_magnitude)
    steep_overlay[steep_mask] = 1
    ax2.imshow(steep_overlay, origin='lower', cmap='Reds', alpha=0.3)
    
    ax2.set_title('Slope Magnitude (Steep Areas Highlighted)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Complete pipeline executed in {z_time + slope_time:.4f} seconds")
    print("Example 5 completed!\n")


if __name__ == "__main__":
    # Run all examples
    example_grid_computation()
    example_compare_grid_methods()
    example_normal_computation()
    
    # Only run FAISS example if the right libraries are available
    if torch.cuda.is_available():
        try:
            import faiss
            example_large_point_cloud_with_faiss()
        except ImportError:
            print("Skipping FAISS example due to missing libraries")
    
    example_workflow_pipeline()
    
    print("All examples completed successfully!") grid Z-values using NumPy
    print("Computing grid Z-values using NumPy...")
    start_time = time.time()
    grid_info['zz'] = compute_grid_z_numpy(point_cloud, grid_info, radius)
    elapsed = time.time() - start_time
    print(f"Computation took {elapsed:.4f} seconds")
    
    # Compute slopes
    print("Computing slopes...")
    slope_x, slope_y = compute_slopes_advanced(grid_info)
    
    # Visualize results
    print("Visualizing results...")
    visualize_results(point_cloud, grid_info, slope_x, slope_y)
    
    print("Example 1 completed!\n")


# ===== Example 2: Comparing Grid Computation Methods =====

def example_compare_grid_methods():
    """
    Compares different grid computation methods.
    """
    print("\n===== Example 2: Comparing Grid Computation Methods =====")
    
    # Parameters
    n_points = 200000     # 200K points
    step_size = 0.1       # 100mm grid step size (coarser for speed)
    radius = 0.05         # 50mm search radius
    x_range = (0, 3)      # X range for terrain
    y_range = (0, 3)      # Y range for terrain
    
    # Generate a synthetic terrain point cloud
    print(f"Generating terrain point cloud with {n_points} points...")
    point_cloud = generate_terrain_point_cloud(n_points, x_range, y_range)
    
    # Create a grid over the terrain
    print(f"Creating grid with {step_size}m step size...")
    grid_info = create_grid_from_extremes(point_cloud, step_size)
    
    # Define methods to test
    methods = {
        "NumPy": compute_grid_z_numpy,
        "NumPy Parallel": lambda pc, gi, r: compute_grid_z_numpy_parallel(pc, gi, r, n_jobs=4)
    }
    
    # Add PyTorch method if available
    if torch.cuda.is_available():
        methods["PyTorch"] = compute_grid_z_torch
        
        # Add FAISS method if available
        try:
            import faiss
            methods["FAISS"] = compute_grid_z_faiss
        except ImportError:
            print("FAISS library not available. Skipping FAISS method.")
    else:
        print("CUDA not available. Skipping GPU methods.")
    
    # Test each method
    results = {}
    for name, method in methods.items():
        print(f"\nTesting {name} method...")
        
        # Warmup run
        _ = method(point_cloud, grid_info, radius)
        
        # Timed run
        start_time = time.time()
        grid_info['zz'] = method(point_cloud, grid_info, radius)
        elapsed = time.time() - start_time
        
        print(f"{name} method took {elapsed:.4f} seconds")
        results[name] = elapsed
    
    # Display comparison
    print("\nMethod comparison:")
    baseline = results["NumPy"]  # Use NumPy as baseline
    for name, time_taken in results.items():
        speedup = baseline / time_taken
        speedup_text = f"{speedup:.2f}x faster than NumPy" if name != "NumPy" else "baseline"
        print(f"  {name}: {time_taken:.4f}s ({speedup_text})")
    
    # Compute slopes with NumPy method
    print("\nComputing slopes...")
    slope_x, slope_y = compute_slopes_advanced(grid_info)
    
    # Visualize results from the last method used
    print("Visualizing results...")
    visualize_results(point_cloud, grid_info, slope_x, slope_y)
    
    print("Example 2 completed!\n")


# ===== Example 3: Normal Computation =====

def example_normal_computation():
    """
    Demonstrates normal computation for a hemisphere.
    """
    print("\n===== Example 3: Normal Computation =====")
    
    # Parameters
    n_points = 5000       # 5K points in the hemisphere
    k_neighbors = 20      # Number of neighbors for normal computation
    grid_size = 20        # Size of the grid for aggregated normals
    
    # Generate synthetic hemisphere data
    print(f"Generating hemisphere with {n_points} points...")
    point_cloud, true_normals = generate_hemisphere(n_points)
    
    # Compute normals using NumPy (typically fastest for this size)
    print("Computing normals using NumPy...")
    start_time = time.time()
    normals = compute_normals_numpy(point_cloud, k=k_neighbors)
    elapsed = time.time() - start_time
    print(f"Normal computation took {elapsed:.4f} seconds")
    
    # Generate grid for aggregated normals
    print("Generating grid for normal aggregation...")
    grid_centers, valid_mask = generate_grid_centers([-1, 1], [-1, 1], grid_size)
    
    # Compute grid normals
    print("Computing grid normals...")
    start_time = time.time()
    grid_normals = compute_grid_normals_numpy(point_cloud, normals, grid_centers, radius=0.2)
    elapsed = time.time() - start_time
    print(f"Grid normal computation took {elapsed:.4f} seconds")
    
    # Plot results
    print("Plotting results...")
    plot_results(point_cloud, normals, grid_centers, grid_normals)
    
    # Evaluate accuracy
    dot_products = np.abs(np.sum(normals * true_normals, axis=1))
    mean_alignment = np.mean(dot_products)
    print(f"Mean normal alignment with true normals: {mean_alignment:.4f} (1.0 is perfect)")
    
    print("Example 3 completed!\n")


# ===== Example 4: Using FAISS for Large Point Clouds =====

def example_large_point_cloud_with_faiss():
    """
    Demonstrates using FAISS for large point cloud processing.
    """
    print("\n===== Example 4: Processing Large Point Cloud with FAISS =====")
    
    # Check if FAISS is available
    try:
        import faiss
    except ImportError:
        print("FAISS library not available. Skipping this example.")
        return
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping this example.")
        return
    
    # Parameters for large point cloud
    n_points = 1000000    # 1M points
    step_size = 0.1       # 100mm grid step size
    radius = 0.05         # 50mm search radius
    x_range = (0, 5)      # Larger X range for bigger terrain
    y_range = (0, 5)      # Larger Y range
    
    # Generate a large synthetic terrain point cloud
    print(f"Generating large terrain point cloud with {n_points:,} points...")
    point_cloud = generate_terrain_point_cloud(n_points, x_range, y_range)
    
    # Create a grid over the terrain
    print(f"Creating grid with {step_size}m step size...")
    grid_info = create_grid_from_extremes(point_cloud, step_size)
    
    # Process with FAISS
    print("Computing grid Z-values using FAISS...")
    start_time = time.time()
    grid_info['zz'] = compute_grid_z_faiss(point_cloud, grid_info, radius)
    elapsed = time.time() - start_time
    print(f"FAISS computation took {elapsed:.4f} seconds")
    
    # Compare with NumPy parallel
    print("Computing grid Z-values using NumPy Parallel for comparison...")
    start_time = time.time()
    zz_numpy = compute_grid_z_numpy_parallel(point_cloud, grid_info, radius, n_jobs=-1)
    elapsed = time.time() - start_time
    print(f"NumPy Parallel computation took {elapsed:.4f} seconds")
    
    # Check similarity of results
    diff = np.abs(grid_info['zz'] - zz_numpy)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    print(f"Maximum difference between methods: {max_diff:.6f}")
    print(f"Mean difference between methods: {mean_diff:.6f}")
    
    # Compute