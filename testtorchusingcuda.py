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