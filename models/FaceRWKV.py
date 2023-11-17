import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from RWKV import Block

class FaceRWKV(nn.Module):
    def __init__(self, config, patch_size, embed_dim, n_layers, n_classes):
        """
        A Face Recognition model utilizing RWKV blocks.

        Args:
        - config: Configuration for the model.
        - patch_size: Size of patches to be extracted from input images.
        - embed_dim: Dimension of the embedded representation.
        - n_layers: Number of RWKV blocks.
        - n_classes: Number of output classes.

        """
        super().__init__()
        self.config = config
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_classes = n_classes

        # Linear projection for the patches
        self.linear_projection = nn.Linear(self.patch_size**2 * 3, self.embed_dim)

        # RWKV Blocks
        self.blocks = nn.Sequential(*[Block(config, i) for i in range(self.n_layers)])

        # MLP Head
        self.mlp_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.n_classes)
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
        - x: Input images, shape (batch_size, 3, H, W).

        Returns:
        - Predictions for classes, shape (batch_size, n_classes).

        """
        # x.shape = (batch_size, 3, H, W)
        # Reshape to patches
        x = image_to_patches(x, self.patch_size)
        # Flatten for linear layer
        x = x.flatten(2)
        x = self.linear_projection(x)
        # x.shape = (batch_size, 14*14, embed_dim)
        x = self.blocks(x)
        # Extract last hidden state
        # x.shape = (batch_size, n_patches, embed_dim) -> (batch_size, embed_dim)
        x = x[:, -1, :]
        # x.shape = (batch_size, n_classes)
        x = self.mlp_head(x)
        return x

def image_to_patches(input_image, patch_size):
    """
    Convert input images into patches of defined size.

    Args:
    - input_image: Input image tensor, shape (batch_size, C, H, W).
    - patch_size: Size of patches to extract.

    Returns:
    - Patches reshaped, shape (batch_size, C, n_patches, patch_size, patch_size).

    """
    N, C, H, W = input_image.shape
    P = patch_size
    assert H % P == 0 and W % P == 0, "Image dimensions should be divisible by patch size."

    patches = input_image.unfold(2, P, P).unfold(3, P, P)
    patches = patches.contiguous().view(N, C, -1, P, P)
    patches = patches.permute(0, 2, 1, 3, 4).contiguous()

    return patches