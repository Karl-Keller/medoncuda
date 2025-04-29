# First run just the NumPy part to test if it works
import os
os.add_dll_directory("C:/Users/karl.keller/AppData/Local/anaconda3/envs/cv2vision/Library/bin")
import numpy as np
from scipy.spatial import KDTree
print("NumPy and SciPy imported successfully")

# Then in a separate script, test the PyTorch part
import torch
print("PyTorch imported successfully")