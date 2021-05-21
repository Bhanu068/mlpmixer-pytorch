import einops
import torch.nn as nn

class MlpBlock(nn.Module):
    """ Multiplayer perceptron.
    
    Parameters
    ----------
    dim : int
        input and output dimension
    
    mlp_dim: int
        hidden layer dimension
    
    """
    def __init__(self, dim, mlp_dim = None):
        super().__init__()
        
        mlp_dim = dim if mlp_dim is None else dim
        self.linear_1 = nn.Linear(dim, mlp_dim)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(mlp_dim, dim)
        
    def forward(self, x):
        """
        x : (torch.Tensor) The input to the model. Shape -> (batch_size, channels, patches) or (batch_size, patches, channels)
        
        Returns:
        torch.Tensor--> Output tensor has exactly the same shape as the input 'x'.
        
        """
        x = self.linear_1(x) # (batch_size, channels, patches)
        x = self.activation(x)
        x = self.linear_2(x)
        
    
class MixerBlock(nn.Module):
    """
    Mixer block that contains channel mixing and token mixing blocks
    
    Parameters
    ----------
    n_pacthes: Number of patches into which the input image is split
    hidden_dim: Dimensionality of patch embeddings
    tokens_mix_dim: Hidden dimensions while performing token mixing
    channels_mix_dim: Hidden dimensions while performing the channel mixing
    
    """
    def __init__(self, *, n_patches, hidden_dim, tokens_mix_dim, channels_mix_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.tokens_mixing_block = MlpBlock(n_patches, tokens_mix_dim)
        self.channels_mixing_block = MlpBlock(hidden_dim, channels_mix_dim)
        
    def forward(self, x):
        """
        x : (torch.Tensor) The input to the model. Shape -> (batch_size, channels, patches) or (batch_size, patches, channels)
        
        Returns:
        torch.Tensor--> Output tensor that has exactly the same shape as the input 'x' 
        
        """
        
        y = self.norm_1(x)
        y = y.permute(0, 2, 1)
        y = self.tokens_mixing_block(y)
        y = y.permute(0, 2, 1)
        x = x + y
        y = self.norm2(x)
        res = x + self.channels_mixing_block(y)
        return res
    
class MlpMixer(nn.Module):
    """
    
    Parameters
    ----------
    img_size: height and width of the input image
    patch_size: height and width of the patches. img_size % patch_size should be 0
    tokens_mix_dim: hidden dim of the mlp block for token mixing
    channels_mix_dim: hidden dim of the mlp block for channel mixing
    num_classes: output classes
    hidden_dim: dimensions of the patch embedding
    num_blocks: The total number of blocks for mixing.
    
    """
    def __init__(self, *, img_size, patch_size, tokens_mix_dim, channels_mix_dim, num_classes, hidden_dim, num_blocks):
        super(MlpMixer, self).__init__()
        num_patchs = img_size // patch_size
        
        self.patch_embedding = nn.Conv2d(3, hidden_dim, kernel_size = patch_size, stride = patch_size)
        self.blocks = nn.ModuleList([
            MixerBlock(n_patches = num_patchs, hidden_dim = hidden_dim, tokens_mix_dim = tokens_mix_dim, 
                       channels_mix_dim = channels_mix_dim) for _ in range(num_blocks)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        """
        
        x : (torch.Tensor) The input to the model. Shape -> (batch_size, channels, patches) or (batch_size, patches, channels)
        
        Returns:
        torch.Tensor--> Output tensor that has exactly the same shape as the input 'x' 
        
        Returns: logits of shape (batch_size, num_classes)

        """
        
        x = self.patch_embedding(x) # shape --> # (batch_size, hidden_dim, num_patches ** (1/2), num_patches ** (1/2))
        x = einops.rearrange(x, "n c h w -> n (h w) c") # shape --> (num_smaples, num_patches, hidden_dim)
        for mixer_block in self.blocks:
            x = mixer_block(x)
        
        x = self.norm(x) # shape --> (num_samples, num_patches, hidden_dim)
        x = x.mean(dim = 1) # shape --> (num_samples, hidden_dim)
        x = self.classifier(x) # shape --> (num_smaples, num_classes) final output
        
        return x # final output
                