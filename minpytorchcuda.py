import os
os.add_dll_directory("C:/Users/karl.keller/AppData/Local/anaconda3/envs/cv2vision/Library/bin")

import torch
# Create a simple tensor on GPU
x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
y = x * 2
print(y)