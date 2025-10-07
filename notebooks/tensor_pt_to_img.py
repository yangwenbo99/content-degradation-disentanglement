import torch
import numpy as np
from PIL import Image
from pathlib import Path

def tensor_to_png(tensor_path, output_path):
    """
    Load a PyTorch tensor from .pt file and save it as a PNG image.
    Assumes the tensor is in range [-1, 1] for normalization.
    """
    # Load the tensor
    tensor = torch.load(tensor_path)
    
    print(f"Loaded tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    print(f"Tensor range: [{tensor.min():.3f}, {tensor.max():.3f}]")
    
    # Convert tensor to numpy array
    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().numpy()
    else:
        array = tensor.numpy()
    
    if array.ndim == 4:  # Shape: (C, H, W)
        array = array[0]  # Take the first element if batch dimension exists

    # Handle different tensor shapes
    if array.ndim == 3:  # Shape: (C, H, W)
        array = np.transpose(array, (1, 2, 0))  # Convert to (H, W, C)
    elif array.ndim == 2:  # Shape: (H, W)
        pass  # Keep as is
    else:
        raise ValueError(f"Unsupported tensor shape: {array.shape}")
    
    # Normalize from [-1, ] to [0, 255]
    array = (array + 1.0) / 2.0 * 255.0
    array = np.clip(array, 0, 255)
    
    # Convert to uint8
    array = array.astype(np.uint8)
    
    # Handle different channel counts
    if array.ndim == 3:
        if array.shape[2] == 3:  # RGB
            img = Image.fromarray(array, 'RGB')
        elif array.shape[2] == 1:  # Grayscale
            img = Image.fromarray(array[:, :, 0], 'L')
        else:
            # Convert to RGB if unknown number of channels
            img = Image.fromarray(array[:, :, :3], 'RGB')
    else:  # Grayscale
        img = Image.fromarray(array, 'L')
    
    # Save the image
    img.save(output_path)
    print(f"Saved image to: {output_path}")

if __name__ == "__main__":
    # Paths
    tensor_path = Path("x_0_tensor.pt")
    output_path = Path("x_0_image.png")
    
    # Check if tensor file exists
    if not tensor_path.exists():
        print(f"Error: {tensor_path} does not exist!")
        exit(1)
    
    # Convert and save
    tensor_to_png(tensor_path, output_path)
