import os

import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from tqdm import tqdm

from explainers.base import Explainer
from utils import device


class RISEExplainer(Explainer):
    def get_saliency(self, img_tensor: torch.Tensor) -> np.ndarray:
        print("Generating RISE Explanation")
        # Initialize RISE with your model
        input_size = (224, 224)  # The input size used during training
        # input_size = (192, 192)  # For when using Resnet18
        gpu_batch = 256  # Use maximum number that the GPU allows.
        explainer = RISE(self.model, input_size, gpu_batch).to(device)

        # Parameters for mask generation
        N = 6000
        s = 8
        p1 = 0.1

        # Generate a filename based on parameters
        filename = f'masks/masks_N={N}_s={s}_p1={p1}_size={input_size[0]}x{input_size[1]}.npy'

        # Generate or load masks
        # Check if masks file exists with the specific configuration
        if not os.path.isfile(filename):
            # If the file doesn't exist, generate new masks and save them
            explainer.generate_masks(N=N, s=s, p1=p1, savepath=filename)
            print(f'New masks generated and saved as {filename}')
        else:
            # If the file exists, load the masks
            explainer.load_masks(filename)
            print(f'Masks loaded from {filename}.')

        return explainer(img_tensor).cpu().numpy()  # Generate saliency map


class RISE(nn.Module):
    def __init__(self, model, input_size, gpu_batch=100):
        super(RISE, self).__init__()
        self.p1 = None
        self.N = None
        self.masks = None
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch

    def generate_masks(self, N, s, p1, savepath='masks.npy'):
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size))

        for i in tqdm(range(N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.cuda()
        self.N = N
        self.p1 = p1

    def load_masks(self, filepath):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float().cuda()
        self.N = self.masks.shape[0]
        self.p1 = get_p1(filepath)

    def forward(self, x):
        N = self.N
        _, _, H, W = x.size()
        # Apply array of filters to the image
        stack = torch.mul(self.masks, x.data)

        # p = nn.Softmax(dim=1)(model(stack)) processed in batches
        p = []
        for i in tqdm(range(0, N, self.gpu_batch), desc='Generating Explanation'):
            p.append(self.model(stack[i:min(i + self.gpu_batch, N)]))
        p = torch.cat(p)
        # Number of classes
        CL = p.size(1)
        sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W))
        sal = sal.view((CL, H, W))
        sal = sal / N / self.p1
        return sal


def get_p1(filepath):
    # Split the filename on underscores
    parts = filepath.split('_')

    # Initialize p1 with a error value in case it's not found
    p1_value = None

    # Search for the part with p1
    for part in parts:
        if part.startswith('p1='):
            # Extract everything after 'p1=' and convert to float
            p1_value = float(part[3:])
            break

    # If p1_value is not None, then we've found and extracted it
    if p1_value is not None:
        return p1_value
    else:
        raise ValueError(f"Failed to extract 'p1' value from the filename: {filepath}")
