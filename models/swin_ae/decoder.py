from models.swin_ae.modules import *

class SwinTransformerDecoder(nn.Module):
    def __init__(self, latent_channels=4):
        super().__init__()

        # Learnable spatial adjustment: 64x64 -> 56x56
        self.spatial_adjust = nn.Conv2d(
            latent_channels, latent_channels,
            kernel_size=9, stride=1, padding=0  # 64 - 9 + 1 = 56
        )

        # Project channels: latent_channels -> 192
        self.from_latent = nn.Sequential(
            nn.Linear(latent_channels, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 192),
            nn.LayerNorm(192),

        )

        self.Stage2 = AlternatingEncoderBlock(192, 6)
        self.Stage1 = AlternatingEncoderBlock(96, 3)

        self.PatchExpanding2 = PatchExpand(192)  # 56x56 -> 112x112, 96ch

        self.output = SwinOutput(in_channels=96)

    def forward(self, x):
        # x: [B, 4, 64, 64]

        x = self.spatial_adjust(x)               # [B, 4, 56, 56] (learnable!)

        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)  # [B, 3136, 4]
        x = self.from_latent(x)                  # [B, 3136, 192]

        x = self.Stage2(x)
        x = self.PatchExpanding2(x)              # [B, 12544, 96]
        x = self.output(self.Stage1(x))          # [B, 3, 448, 448]

        return x

# class SwinTransformerDecoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.Stage1 = AlternatingEncoderBlock(96, 3)
#         self.Stage2 = AlternatingEncoderBlock(192, 6)
#         self.Stage3_1 = AlternatingEncoderBlock(384, 12)
#         self.Stage3_2 = AlternatingEncoderBlock(384, 12)
#         self.Stage3_3 = AlternatingEncoderBlock(384, 12)
#         self.Stage4 = AlternatingEncoderBlock(768, 24)

#         self.PatchExpanding4 = PatchExpand(768)
#         self.PatchExpanding3 = PatchExpand(384)
#         self.PatchExpanding2 = PatchExpand(192)

#         self.output = SwinOutput(in_channels=96)

#         self.ReshapeTokenDim = nn.Sequential(
#             nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=0),  # Channel expansion only
#             nn.BatchNorm2d(512),
#         )

#         self.IncreaseChannel = nn.Sequential(
#             nn.Linear(512, 768),
#             nn.LayerNorm(768),
#         )

#     def forward(self, x):
#         x = self.ReshapeTokenDim(x)  # [B, 64, 16, 16] → [B, 512, 14, 14]
#         B, C, H, W = x.shape
#         x = x.view(B, C, H * W).transpose(1, 2) # [B, 196, 512]
#         x = self.IncreaseChannel(x)

#         x = self.PatchExpanding4(self.Stage4(x))
#         x = self.Stage3_1(x)
#         x = self.Stage3_2(x)
#         x = self.Stage3_3(x)
#         x = self.PatchExpanding3(x)
#         x = self.PatchExpanding2(self.Stage2(x))
#         x = self.output(x) # [B, 3, 448, 448]



#         # print(f"🐛 Input dtype: {x.dtype}, shape: {x.shape}")

#         # x = self.ReshapeTokenDim(x)
#         # print(f"🐛 After ReshapeTokenDim: {x.dtype}")

#         # B, C, H, W = x.shape
#         # x = x.view(B, C, H * W).transpose(1, 2)
#         # x = self.IncreaseChannel(x)
#         # print(f"🐛 After IncreaseChannel: {x.dtype}")

#         # x = self.PatchExpanding4(self.Stage4(x))
#         # print(f"🐛 After Stage4+PatchExpanding4: {x.dtype}")

#         # x = self.Stage3_1(x)
#         # print(f"🐛 After Stage3_1: {x.dtype}")

#         # x = self.Stage3_2(x)
#         # print(f"🐛 After Stage3_2: {x.dtype}")

#         # x = self.Stage3_3(x)
#         # print(f"🐛 After Stage3_3: {x.dtype}")

#         # x = self.PatchExpanding3(x)
#         # print(f"🐛 After PatchExpanding3: {x.dtype}")

#         # x = self.PatchExpanding2(self.Stage2(x))
#         # print(f"🐛 After Stage2+PatchExpanding2: {x.dtype}")

#         # x = self.output(x)
#         # print(f"🐛 After output: {x.dtype}")

#         return x