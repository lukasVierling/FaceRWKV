import torch
import torch.nn as nn
import torchvision.models as models

class LinearSequencing(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super(LinearSequencing, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.linear = nn.Linear(patch_size**2 * 3, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.image_to_patches(x)
        # Flatten for linear layer
        x = x.flatten(2)
        x = self.linear(x)
        x = self.layer_norm(x)
        return x
    
    def image_to_patches(self, input_image):
        """
        Convert input images into patches of defined size.

        Args:
        - input_image: Input image tensor, shape (batch_size, C, H, W).
        - patch_size: Size of patches to extract.

        Returns:
        - Patches reshaped, shape (batch_size, C, n_patches, patch_size, patch_size).

        """
        N, C, H, W = input_image.shape
        P = self.patch_size
        assert H % P == 0 and W % P == 0, "Image dimensions should be divisible by patch size."

        patches = input_image.unfold(2, P, P).unfold(3, P, P)
        patches = patches.contiguous().view(N, C, -1, P, P)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()

        return patches
    
class CNNSequencing(nn.Module):
    def __init__(self, patch_size, embed_dim): #patch_size basically irrelevant here
        super(CNNSequencing, self).__init__()
        # load a pretrained resnet and up to stage 3
        self.resnet = models.resnet50(pretrained=True)
        # Remove the layers after stage 4
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:4])
        #for param in self.resnet.parameters():
         #   param.requires_grad = False
            
        self.conv = nn.Conv2d(64, embed_dim, kernel_size=5, stride=5)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.patch_size = patch_size
        self.embed_dim = embed_dim

    def forward(self, x):
        x = self.resnet(x)
        x = self.conv(x)
        #output shape is: [batch_size, 2048, 12, 18]
        N, C, H, W = x.size()
        x = x.view(N, C, -1).permute(0, 2, 1)
        x = self.layer_norm(x)
        return x

