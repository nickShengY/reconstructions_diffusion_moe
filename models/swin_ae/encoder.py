from models.swin_ae.modules import *


class SwinTransformer(nn.Module):
    def __init__(self, latent_channels=4):
        super().__init__()
        # 448/4 = 112, then 112/2 = 56 after one merge
        # We need 64x64, so we use a conv to adjust

        self.Embedding = SwinEmbedding(C=96)   # 448 -> 112x112, 96ch
        self.PatchMerge1 = PatchMerging(96)    # 112 -> 56x56, 192ch

        self.Stage1 = AlternatingEncoderBlock(96, 3)
        self.Stage2 = AlternatingEncoderBlock(192, 6)

        # Project channels: 192 -> latent_channels
        self.to_latent = nn.Sequential(
            nn.Linear(192, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, latent_channels),
            nn.LayerNorm(latent_channels),
        )

        # Learnable spatial adjustment: 56x56 -> 64x64
        self.spatial_adjust = nn.ConvTranspose2d(
            latent_channels, latent_channels,
            kernel_size=9, stride=1, padding=0  # 56 + 9 - 1 = 64
        )

    def forward(self, x):
        x = self.Embedding(x)                  # [B, 12544, 96]
        x = self.PatchMerge1(self.Stage1(x))   # [B, 3136, 192]
        x = self.Stage2(x)                     # [B, 3136, 192]

        x = self.to_latent(x)                  # [B, 3136, 4]

        B, N, C = x.shape
        H = W = int(N**0.5)                    # 56
        x = x.transpose(1, 2).view(B, C, H, W) # [B, 4, 56, 56]

        x = self.spatial_adjust(x)             # [B, 4, 64, 64] (learnable!)

        return x

# class SwinTransformer(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.Embedding = SwinEmbedding(C=96)
#         self.PatchMerge1 = PatchMerging(96)
#         self.PatchMerge2 = PatchMerging(192)
#         self.PatchMerge3 = PatchMerging(384)
#         self.Stage1 = AlternatingEncoderBlock(96, 3)
#         self.Stage2 = AlternatingEncoderBlock(192, 6)
#         self.Stage3_1 = AlternatingEncoderBlock(384, 12)
#         self.Stage3_2 = AlternatingEncoderBlock(384, 12)
#         self.Stage3_3 = AlternatingEncoderBlock(384, 12)
#         self.Stage4 = AlternatingEncoderBlock(768, 24)

#         self.ReduceChannel = nn.Sequential(
#             nn.Linear(768, 512),
#             nn.LayerNorm(512),
#         )

#         self.ReshapeSpatialDim = nn.Sequential(
#             nn.ConvTranspose2d(512, 128, kernel_size=3, stride=1, padding=0),
#             nn.BatchNorm2d(128),
#         )

#     def forward(self, x):
#         x = self.Embedding(x)
#         x = self.PatchMerge1(self.Stage1(x))
#         x = self.PatchMerge2(self.Stage2(x))
#         x = self.Stage3_1(x)
#         x = self.Stage3_2(x)
#         x = self.Stage3_3(x)
#         x = self.PatchMerge3(x)
#         x = self.Stage4(x) # (B, 49, 768) token x channels

#         x = self.ReduceChannel(x)
#         B, N, C = x.shape
#         H = W = int(N**0.5)
#         x = x.transpose(1, 2).view(B, C, H, W)  # (B, 256, H, W)
#         x = self.ReshapeSpatialDim(x)  # (B, 64, 16, 16)

#         return x