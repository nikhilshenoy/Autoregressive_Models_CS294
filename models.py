import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

class MaskConv2D(nn.Conv2d):
    def __init__(self, mask_type, *args, color_conditioning = False, conditional_size = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', torch.zeros(self.weight.size())) # Weight shape is same as tensor shape
        self.conditional_size = conditional_size
        assert mask_type in ['A', 'B'], "Unknown Mask Type"
        h =  self.kernel_size[0]
        w = self.kernel_size[1]
        
        # Creating masks for autoregressive properties
        self.mask[:, :, :h//2, :] = 1  # Mask type A 
        self.mask[:, :, h//2, :w//2 + (mask_type == 'B')] = 1 # Mask type B
        
        # Adding autoregressive property of color channels
        if color_conditioning:
            in_third, out_third = self.in_channels // 3, self.out_channels // 3
            if mask_type == 'B':
                self.mask[2*out_third:, :, h // 2, w // 2] = 1 # B has connections from R, G and B of input mask
                self.mask[out_third:2*out_third, :2*in_third, h // 2, w // 2] = 1 # G has connections from R and G
                self.mask[out_third:, in_third:, h // 2, w // 2] = 1 # R has connections only from R
            else:
                self.mask[out_third:2*out_third, :in_third, h // 2, w // 2] = 1  # G has connections from R
                self.mask[2*out_third:, :2*in_third, h // 2, w // 2] = 1 # B has connections from R and G
                
        if self.conditional_size:
            if len(self.conditional_size) == 1:
                self.cond_op = nn.Linear(conditional_size[0], self.out_channels)
            else:
                self.cond_op = nn.Conv2d(conditional_size[0], self.out_channels, stride = 1,
                                         kernel_size = 3, padding = 1)
      
    def forward(self, x, cond = None):
        self.weight.data *= self.mask
        out = super(MaskConv2D, self).forward(x)
        if self.conditional_size:
            if len(self.conditional_size) == 1:
                out = out + self.cond_op(cond).view(x.shape[0], -1, 1, 1)
            else:
                out = out + self.cond_op(cond)
        return out

class ResBlock(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(ResBlock, self).__init__()
        self.net = nn.ModuleList()
        self.net = self.net.extend([
            nn.ReLU(),
            MaskConv2D('B', in_channels, in_channels // 2, 1, 1, 1 // 2, **kwargs),
            nn.ReLU(),
            MaskConv2D('B', in_channels // 2, in_channels // 2, 7, 1,  7 // 2, **kwargs),
            nn.ReLU(),
            MaskConv2D('B', in_channels // 2, in_channels, 1, 1, 1 // 2, **kwargs),
        ])
    def forward(self, x, cond = None):
        out = x
        for layer in self.net:
            if isinstance(layer, MaskConv2D):
                out = layer(out, cond = cond)
            else:
                out = layer(out)
        return x + out


class PixelCNN(nn.Module):
    def __init__(self, input_shape, channels, colors, no_of_layers,
                 color_conditioning, use_ResBlock, conditional_size = None, device = None):
        super(PixelCNN, self).__init__()
        self.input_shape = input_shape
        self.device = device
        self.channels = channels
        self.color_channels = colors
        self.color_conditioning = color_conditioning
        self.conditional_size = conditional_size
        
        # Define kwargs based on input
        kwargs = dict(
            color_conditioning = self.color_conditioning,
            conditional_size = self.conditional_size
        )
        
        # Initialize block function to be used repeatedly
        if use_ResBlock:
            block = lambda: ResBlock(channels, **kwargs)    
        else:
            block = lambda: MaskConv2D('B', channels, channels,
                                       kernel_size = 7, padding = 7 // 2,
                                       **kwargs)
        
        # 7 x 7 Conv2D operation using Type A Mask
        kernel_size = 7
        self.net = nn.ModuleList()
        self.net.extend([MaskConv2D('A', input_shape[0], channels,
                                    kernel_size = 7, padding = 7 // 2,
                                    **kwargs)])
        
        # 5 7 x 7 Conv2D operation using Type B Mask
        for _ in range(no_of_layers):
            self.net.extend([nn.ReLU(),
                                nn.BatchNorm2d(channels),
                                block(),])
        
        # 2 1 x 1 Conv2D operation using Type B Mask
        self.net.extend([nn.ReLU(),
                            nn.BatchNorm2d(channels),
                            MaskConv2D('B', channels, channels, 1, 1, **kwargs),
                            nn.ReLU(),
                            nn.BatchNorm2d(channels),
                            MaskConv2D('B', channels, self.color_channels * self.input_shape[0],
                                       1, 1, **kwargs)])
        
        if self.conditional_size:
            if len(self.conditional_size) == 1:
                self.cond_op = lambda x: x  # identity
            else:
                self.cond_op = nn.Sequential(
            F.relu(nn.Conv2d(1, 64, 3, padding=1)),
            F.relu(nn.Conv2d(64, 64, 3, padding=1)),
            F.relu(nn.Conv2d(64, 64, 3, padding=1)),
        )

    def forward(self, x, cond = None):
        batch_size = x.shape[0]
        out = (x.float() / (self.color_channels - 1) - 0.5) / 0.5
        if self.conditional_size:
            cond = self.cond_op(cond)
        for layer in self.net:
            if isinstance(layer, MaskConv2D) or isinstance(layer, ResBlock):
                out = layer(out, cond)
            else:
                out = layer(out)
            
        if self.color_conditioning:
            return out.view(batch_size, self.input_shape[0], self.color_channels,
                          *self.input_shape[1:]).permute(0, 2, 1, 3, 4)
        else:
            return out.view(batch_size, self.color_channels, *self.input_shape)
    
    def loss(self, x, cond = None):
        return F.cross_entropy(self(x, cond = cond), x.long())
 
    def get_samples(self, n, cond=None):
        samples = torch.zeros([n, *self.input_shape]).to(self.device)
        with torch.no_grad():
            for r in range(self.input_shape[1]):
                for c in range(self.input_shape[2]):
                    for k in range(self.input_shape[0]):
                        out = self(samples, cond=cond)[:, :, k, r, c]
                        probs = F.softmax(out, dim = 1)
                        samples[:, k, r, c] = torch.multinomial(probs, 1).squeeze(-1)
        return samples.permute(0, 2, 3, 1).cpu().numpy()