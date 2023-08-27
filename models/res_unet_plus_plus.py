import torch.nn as nn
import torch

class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class AttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim):
        super(AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_decoder),
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2                


class ResUnetPlusPlus(nn.Module):
    def __init__(self, channel, filters=[32, 64, 128, 256, 512]):
        super(ResUnetPlusPlus, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])
        
        self.residual_conv_block_1 = nn.Sequential(
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(),
            nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1),
        )
        
        self.conv_skip_1 = nn.Sequential(
            nn.Conv2d(filters[0], filters[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters[1]),
        )
        
        self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])
        
        self.residual_conv_block_2 = nn.Sequential(
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(),
            nn.Conv2d(filters[1], filters[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(),
            nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1),
        )
        
        self.conv_skip_2 = nn.Sequential(
            nn.Conv2d(filters[1], filters[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters[2]),
        )

        self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])
        
        self.residual_conv_block_3 = nn.Sequential(
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(),
            nn.Conv2d(filters[2], filters[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(),
            nn.Conv2d(filters[3], filters[3], kernel_size=3, padding=1),
        )
        
        self.conv_skip_3 = nn.Sequential(
            nn.Conv2d(filters[2], filters[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters[3]),
        )   

        self.aspp_bridge = ASPP(filters[3], filters[4])

        self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        self.upsample1 = nn.Upsample(mode="bilinear", scale_factor=2)
        
        self.up_residual_conv_block_1 = nn.Sequential(
            nn.BatchNorm2d(filters[4] + filters[2]),
            nn.ReLU(),
            nn.Conv2d(filters[4] + filters[2], filters[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(),
            nn.Conv2d(filters[3], filters[3], kernel_size=3, padding=1),
        )
        
        self.up_conv_skip_1 = nn.Sequential(
            nn.Conv2d(filters[4] + filters[2], filters[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[3]),
        )
        
        self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        self.upsample2 =  nn.Upsample(mode="bilinear", scale_factor=2)
        
        self.up_residual_conv_block_2 = nn.Sequential(
            nn.BatchNorm2d(filters[3] + filters[1]),
            nn.ReLU(),
            nn.Conv2d(filters[3] + filters[1], filters[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(),
            nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1),
        )
        
        self.up_conv_skip_2 = nn.Sequential(
            nn.Conv2d(filters[3] + filters[1], filters[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[2]),
        )
        
        self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])
        self.upsample3 = nn.Upsample(mode="bilinear", scale_factor=2)
        
        self.up_residual_conv_block_3 = nn.Sequential(
            nn.BatchNorm2d(filters[2] + filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[2] + filters[0], filters[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(),
            nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1),
        )
        
        self.up_conv_skip_3 = nn.Sequential(
            nn.Conv2d(filters[2] + filters[0], filters[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[1]),
        )
        
        self.aspp_out = ASPP(filters[1], filters[0])

        self.output_layer = nn.Sequential(nn.Conv2d(filters[0], 1, 1), nn.Sigmoid())

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv_block_1(x2) + self.conv_skip_1(x2)

        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv_block_2(x3) + self.conv_skip_2(x3)

        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv_block_3(x4) + self.conv_skip_3(x4)

        x5 = self.aspp_bridge(x4)

        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv_block_1(x6) + self.up_conv_skip_1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv_block_2(x7) + self.up_conv_skip_2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv_block_3(x8) + self.up_conv_skip_3(x8)

        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)

        return out
