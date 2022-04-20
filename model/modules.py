import config

import torch.nn as nn

CONV_STRIDE = 2
CONV_PADDING = 1 
CONV_KERNEL_SIZE = 3

# General Generator Conv blocks
class EncoderConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, bias):
        super().__init__()
        self.conv_layer = nn.ConvTranspose2d(in_channels=dim_in,
                                    out_channels=dim_out,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    bias=bias)
        self.batch_norm = nn.BatchNorm2d(dim_out, track_running_stats=True)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        output = self.relu(self.batch_norm(self.conv_layer(x)))
        return output

# General Generator Conv blocks
class DecoderConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, bias):
        super().__init__()

        self.conv_layer = nn.Conv2d(in_channels=dim_in,
                                            out_channels=dim_out,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            bias=bias)
        self.batch_norm = nn.BatchNorm2d(dim_out, track_running_stats=True)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.batch_norm(self.conv_layer(x)))

# The final layer of the decoder uses the Tanh activation function
class FinalGeneratorConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, bias):
        super().__init__()

        self.conv_layer = nn.Conv2d(in_channels=dim_in,
                                            out_channels=dim_out,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            bias=bias)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.conv_layer(x))
        return x


class DiscriminatorConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, bias):
        super().__init__()

        self.conv_layer = nn.Conv2d(in_channels=dim_in,
                                    out_channels=dim_out,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    bias=bias)
        self.batch_norm = nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True)
        # InstanceNorm2d TODO try this
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        output = self.relu(self.batch_norm(self.conv_layer(x)))
        return output

class Generator(nn.Module):
    """Generator network."""
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc_1 =  EncoderConvBlock(dim_in=config.conv_filters,
                                dim_out=256,
                                kernel_size=CONV_KERNEL_SIZE,
                                stride=CONV_STRIDE,
                                padding=CONV_PADDING,
                                bias=False)

        self.enc_2 =     EncoderConvBlock(dim_in=256,
                                dim_out=512,
                                kernel_size=CONV_KERNEL_SIZE,
                                stride=CONV_STRIDE,
                                padding=CONV_PADDING,
                                bias=False)

        self.enc_3 =    EncoderConvBlock(dim_in=512,
                                dim_out=1024,
                                kernel_size=CONV_KERNEL_SIZE,
                                stride=CONV_STRIDE,
                                padding=CONV_PADDING,
                                bias=False)
       
        # Decoder
        self.dec_1 =     DecoderConvBlock(dim_in=1024,
                                dim_out=512,
                                kernel_size=CONV_KERNEL_SIZE,
                                stride=CONV_STRIDE,
                                padding=CONV_PADDING,
                                bias=False)

        self.dec_2 =     DecoderConvBlock(dim_in=512,
                                dim_out=256,
                                kernel_size=CONV_KERNEL_SIZE,
                                stride=CONV_STRIDE,
                                padding=CONV_PADDING,
                                bias=False)
        self.dec_3 =    FinalGeneratorConvBlock(dim_in=256,
                                dim_out=config.conv_filters,
                                kernel_size=CONV_KERNEL_SIZE,
                                stride=CONV_STRIDE,
                                padding=CONV_PADDING,
                                bias=False)

    def forward(self, x):
        # Layers are separated for easier debugging
        x = self.enc_1(x)
        x = self.enc_2(x)
        x = self.enc_3(x)
        x = self.dec_1(x)
        x = self.dec_2(x)
        x = self.dec_3(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 =    DiscriminatorConvBlock(dim_in=config.conv_filters,
                                dim_out=256,
                                kernel_size=CONV_KERNEL_SIZE,
                                stride=CONV_STRIDE,
                                padding=CONV_PADDING,
                                bias=False)

        self.d2=    DiscriminatorConvBlock(dim_in=256,
                                dim_out=512,
                                kernel_size=CONV_KERNEL_SIZE,
                                stride=CONV_STRIDE,
                                padding=CONV_PADDING,
                                bias=False)

        self.d3=    DiscriminatorConvBlock(dim_in=512,
                                dim_out=1024,
                                kernel_size=CONV_KERNEL_SIZE,
                                stride=CONV_STRIDE,
                                padding=CONV_PADDING,
                                bias=False)
        # final conv layer
        self.d4 = nn.Conv2d(in_channels=1024,
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False)

    def forward(self, x):
        # Layers are separated for easier debugging
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return x