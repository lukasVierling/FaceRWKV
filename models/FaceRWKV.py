import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from models.RWKV import Block

class FaceRWKV(nn.Module):
    def __init__(self, config):
        """
        A Face Recognition model utilizing RWKV blocks.

        Args:
        - config: Configuration for the model.

        """
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.embed_dim = config.n_embd
        self.n_layers = config.n_layer
        self.n_classes = config.n_classes

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

class RWKVConfig:
    def __init__(self):
        # Model architecture parameters
        self.n_embd = 256           # Embedding size
        self.n_attn = 4             # Number of attention heads
        self.n_head = 4             # Number of heads for RWKV_TinyAttn
        self.ctx_len = 256          # Context length -> apparently crashes for too short context????
        #self.vocab_size = 50000    # Vocabulary size
        self.rwkv_emb_scale = 1.0   # Scale for final projection in RWKV_TimeMix and RWKV_ChannelMix
        self.rwkv_tiny_attn = 64    # Tiny attention size for RWKV_TinyAttn
        self.rwkv_tiny_head = 2     # Number of tiny attention heads for RWKV_TinyAttn
        self.n_ffn = 512            # Hidden size for RWKV_ChannelMix
        self.n_layer = 4            # Number of RWKV blocks
        self.patch_size = 20        # Size of patches to be extracted from input images
        self.n_classes = 7          # Number of output classes

        # Initialization parameters
        self.scale_init = 0  # Scale for weight initialization in RWKV_TimeMix and RWKV_ChannelMix

    def calculate_decay_speed(self, h):
        return math.pow(self.ctx_len, -(h + 1) / (self.n_head - 1))
