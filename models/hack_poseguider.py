import os
import torch
import torch.nn as nn
import torch.nn.init as init
from einops import rearrange
import numpy as np

class Hack_PoseGuider(nn.Module):
    def __init__(self, noise_latent_channels=320):
        super(Hack_PoseGuider, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.SiLU()
        )

        # Final projection layer
        self.final_proj = nn.Conv2d(in_channels=128, out_channels=noise_latent_channels, kernel_size=1)

        # Initialize layers
        self._initialize_weights()

        self.scale = nn.Parameter(torch.ones(1) * 2)

    # def _initialize_weights(self):
    #     # Initialize weights with Gaussian distribution and zero out the final layer
    #     for m in self.conv_layers:
    #         if isinstance(m, nn.Conv2d):
    #             init.normal_(m.weight, mean=0.0, std=0.02)
    #             if m.bias is not None:
    #                 init.zeros_(m.bias)

    #     init.zeros_(self.final_proj.weight)
    #     if self.final_proj.bias is not None:
    #         init.zeros_(self.final_proj.bias)
    
    def _initialize_weights(self):
        # Initialize weights with He initialization and zero out the biases
        for m in self.conv_layers:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                init.normal_(m.weight, mean=0.0, std=np.sqrt(2. / n))
                if m.bias is not None:
                    init.zeros_(m.bias)

        # For the final projection layer, initialize weights to zero (or you may choose to use He initialization here as well)
        init.zeros_(self.final_proj.weight)
        if self.final_proj.bias is not None:
            init.zeros_(self.final_proj.bias)


    def forward(self, x):
        x = self.conv_layers(x)
        x = self.final_proj(x)

        return x * self.scale

    @classmethod
    def from_pretrained(cls,pretrained_model_path):
        if not os.path.exists(pretrained_model_path):
            print(f"There is no model file in {pretrained_model_path}")
        print(f"loaded PoseGuider's pretrained weights from {pretrained_model_path} ...")

        state_dict = torch.load(pretrained_model_path, map_location="cpu")
        model = Hack_PoseGuider(noise_latent_channels=320)
                
        m, u = model.load_state_dict(state_dict, strict=False)
        # print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")        
        params = [p.numel() for n, p in model.named_parameters()]
        print(f"### PoseGuider's Parameters: {sum(params) / 1e6} M")
        
        return model
