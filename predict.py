from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from scipy import io
import argparse
import os

from network import DDPM, UNET

def predict_mnist(digit: Optional[int] = None, n_sample: int = 5, guide_w: float = 2.0, save_name: str = "prediction.png"):
    """
    Generate MNIST digits using the diffusion model.
    
    Args:
        digit: Specific digit to generate (0-9). If None, generates random digits.
        n_sample: Number of samples to generate
        guide_w: Guidance weight for classifier-free guidance (higher = stronger conditioning)
        save_name: Name of the output file
    """
    n_T = 400 # 500
    device = "cpu"
    n_classes = 10
    n_feat = 128
    save_dir = './data/diffusion_output/prediction/'
    path = './data/diffusion_output/model_19.pth'
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    ddpm = DDPM(nn_model=UNET(in_channels=1, n_features=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.load_state_dict(torch.load(path, map_location=torch.device(device)))
    ddpm.to(device)
    ddpm.eval()

    with torch.no_grad():
        # Create context labels
        if digit is not None:
            # Generate specific digit(s)
            if isinstance(digit, int):
                assert 0 <= digit <= 9, "Digit must be between 0 and 9"
                c = torch.tensor([digit] * n_sample, dtype=torch.long).to(device)
                print(f"Generating {n_sample} samples of digit {digit}")
            else:
                # If digit is a list, use it directly
                c = torch.tensor(digit, dtype=torch.long).to(device)
        else:
            # Generate random digits (0-9)
            c = torch.randint(0, 10, (n_sample,), dtype=torch.long).to(device)
            print(f"Generating {n_sample} random digits")
        
        x_gen, x_gen_store = ddpm.prediction(n_sample, (1, 28, 28), c, device, guide_w=guide_w)

        # Save generated images
        grid = make_grid(x_gen*-1 + 1, nrow=n_sample)
        save_path = os.path.join(save_dir, save_name)
        save_image(grid, save_path)
        print(f'Saved image at {save_path}')
        
        return x_gen, c


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate MNIST digits using diffusion model')
    parser.add_argument('--digit', type=int, default=None, 
                        help='Specific digit to generate (0-9). If not specified, generates random digits.')
    parser.add_argument('--n_sample', type=int, default=5,
                        help='Number of samples to generate (default: 5)')
    parser.add_argument('--guide_w', type=float, default=2.0,
                        help='Guidance weight for classifier-free guidance (default: 2.0)')
    parser.add_argument('--save_name', type=str, default='prediction.png',
                        help='Name of the output file (default: prediction.png)')
    
    args = parser.parse_args()
    
    predict_mnist(digit=args.digit, n_sample=args.n_sample, guide_w=args.guide_w, save_name=args.save_name)