import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

# Get the CUDA device name
if cuda_available:
    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name(device)}")
else:
    print("CUDA is not available. Please check your installation.")
