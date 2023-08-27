import torch
import torch.nn as nn

class ResUnet(nn.Module):
    def __init__(self, channel, filters=[64, 128, 256, 512]):
        super(ResUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )
        
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
        

        self.upsample_1 = nn.ConvTranspose2d(filters[3], filters[3], kernel_size=2, stride=2)
        
        self.up_residual_conv_block_1 = nn.Sequential(
            nn.BatchNorm2d(filters[3] + filters[2]),
            nn.ReLU(),
            nn.Conv2d(filters[3] + filters[2], filters[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(),
            nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1),
        )
        
        self.up_conv_skip_1 = nn.Sequential(
            nn.Conv2d(filters[3] + filters[2], filters[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[2]),
        )
        
        self.upsample_2 = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=2, stride=2)
        
        self.up_residual_conv_block_2 = nn.Sequential(
            nn.BatchNorm2d(filters[2] + filters[1]),
            nn.ReLU(),
            nn.Conv2d(filters[2] + filters[1], filters[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(),
            nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1),
        )
        
        self.up_conv_skip_2 = nn.Sequential(
            nn.Conv2d(filters[2] + filters[1], filters[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[1]),
        )
        
        self.upsample_3 = nn.ConvTranspose2d(filters[1], filters[1], kernel_size=2, stride=2)
        
        self.up_residual_conv_block_3 = nn.Sequential(
            nn.BatchNorm2d(filters[1] + filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[1] + filters[0], filters[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        
        self.up_conv_skip_3 = nn.Sequential(
            nn.Conv2d(filters[1] + filters[0], filters[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[0]),
        )    
        
        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_block_1(x1) + self.conv_skip_1(x1)
        x3 = self.residual_conv_block_2(x2) + self.conv_skip_2(x2)
        
        # Bridge
        x4 = self.residual_conv_block_3(x3) + self.conv_skip_3(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv_block_1(x5) + self.up_conv_skip_1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv_block_2(x7) + self.up_conv_skip_2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv_block_3(x9) + self.up_conv_skip_3(x9)

        output = self.output_layer(x10)

        return output
