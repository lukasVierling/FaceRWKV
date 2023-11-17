import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.RWKV import RWKV

#In this class we will implement an architecture that is based on the Vision Transformer architecture but we replace the transformer with an RWKV block

class FaceRWKV(nn.Module):
    def __init__(self, config, patch_size, embed_dim, n_layers, n_classes):
        super().__init__()
        self.config = config
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_classes = n_classes

        # Linear projection for the patches
        self.linear_projection = nn.Linear(self.patch_size**2 * 3, self.embed_dim)

        # RWKV Blocks
        self.blocks = nn.Sequential(*[RWKV(config, i) for i in range(config.n_layer)])

        # MLP Head
        self.mlp_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.n_classes)
        )

    def forward(self, x):
        # x.shape = (batch_size, 3, 224, 224)
        x = self.linear_projection(x)
        # x.shape = (batch_size, 14*14, embed_dim)
        x = self.blocks(x)
        # extract last hidden state
        # x.shape = (batch_size, n_patches, embed_dim) -> (batch_size, embed_dim)
        x = x[:, -1, :]
        # x.shape = (batch_size, n_classes)
        x = self.mlp_head(x)
        return x