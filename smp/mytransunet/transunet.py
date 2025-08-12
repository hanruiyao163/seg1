import torch
from torch import nn
from einops import rearrange, repeat
from vision_transformer import VisionTransformer
import numpy as np

class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, channel_factor=1):
        super().__init__()

        if (stride != 1 or in_channels != out_channels):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        mid_channels = out_channels * channel_factor

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1)
        self.norm2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        x_down = x
        if hasattr(self, 'downsample'):
            x_down = self.downsample(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = x + x_down
        x = self.relu(x)

        return x
    

class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, x_concat=None):
        x = self.upsample(x)

        if x_concat is not None:
            x = torch.cat([x_concat, x], dim=1)

        x = self.layer(x)
        return x
    
class VIT_TORCH(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.vit = VisionTransformer(*args, **kwargs)
        del self.vit.conv_proj
        del self.vit.heads

class Encoder(nn.Module):
    def __init__(self, img_size, in_channels, out_channels, num_heads=8, mlp_dim=512, num_layers=12):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)

        self.vit_img_size = img_size // 16
        self.vit = VisionTransformer(image_size=self.vit_img_size,
                                     patch_size=1,
                                     num_layers=num_layers,
                                     num_heads=num_heads,
                                     hidden_dim=out_channels * 8,
                                     mlp_dim=mlp_dim,
                                     in_channels=out_channels * 8)

        self.conv2 = nn.Conv2d(out_channels * 8, 512, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x1 = self.relu(x)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x = self.encoder3(x3)
        print("encoder", x.shape)

        hidden_states = self.vit(x)
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        print("n_patch", n_patch)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.contiguous().view(B, hidden, h, w)
        print("vit", x.shape)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        return x, x1, x2, x3

class Decoder(nn.Module):
    def __init__(self, out_channels, num_classes):
        super().__init__()

        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder4 = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 8))

        self.conv1 = nn.Conv2d(int(out_channels * 1 / 8), num_classes, kernel_size=1)

    def forward(self, x, x1, x2, x3):
        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x)
        x = self.conv1(x)

        return x
    
class TransUNet(nn.Module):
    def __init__(self, img_size, in_channels, out_channels, num_heads, mlp_dim, num_layers, num_classes):
        super().__init__()

        self.encoder = Encoder(img_size=img_size, in_channels=in_channels, out_channels=out_channels,
                               num_heads=num_heads, mlp_dim=mlp_dim, num_layers=num_layers)

        self.decoder = Decoder(out_channels, num_classes)

    def forward(self, x):
        x, x1, x2, x3 = self.encoder(x)
        x = self.decoder(x, x1, x2, x3)

        return x

if __name__ == "__main__":
    x = torch.randn(1, 3, 512, 512)
    # model = DecoderBottleneck(3, 128)
    # print(model(x).shape)
    model = TransUNet(img_size=512, in_channels=3, out_channels=128, num_heads=8, mlp_dim=512, num_layers=12, num_classes=11)
    y = model(x)
    print(y.shape)