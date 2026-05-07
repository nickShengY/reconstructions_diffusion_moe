from models.swin_ae.encoder import SwinTransformer
from models.swin_ae.decoder import SwinTransformerDecoder
import torch.nn as nn

class SwinAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SwinTransformer()
        self.decoder = SwinTransformerDecoder()

    def forward(self, x):
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out