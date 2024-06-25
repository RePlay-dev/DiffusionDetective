import numpy as np
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_y(path):
    """Custom labeler to categorize images into 'fake' or 'real'."""
    if 'fake' in path.parts:
        return 'fake'
    elif 'real' in path.parts:
        return 'real'
    else:
        raise ValueError(f"Path to image ({str(path)}) contains neither a 'fake' or a 'real' folder")
