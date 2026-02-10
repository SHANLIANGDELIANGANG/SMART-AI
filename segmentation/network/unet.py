import torch
from torch import nn, optim

def kaiming_weight_init(m, bn_std=0.02):

    classname = m.__class__.__name__
    if 'Conv3d' in classname or 'ConvTranspose3d' in classname:
        version_tokens = torch.__version__.split('.')
        if int(version_tokens[0]) == 0 and int(version_tokens[1]) < 4:
            nn.init.kaiming_normal(m.weight)
        else:
            nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif 'BatchNorm' in classname:
        m.weight.data.normal_(1.0, bn_std)
        m.bias.data.zero_()
    elif 'Linear' in classname:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()


class StackedConvLayers(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(StackedConvLayers, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.act(self.bn(self.conv(input)))
        return out


class SegmentationNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SegmentationNet, self).__init__()

        self.encoder_conv_blocks = []
        #encoder
        self.encoder_conv_blocks.append(
            StackedConvLayers(in_channels, 28)
        )
        self.encoder_conv_blocks.append(
            StackedConvLayers(28, 28),
        )
        self.encoder_conv_blocks.append(
            StackedConvLayers(28, 60)
        )
        self.encoder_conv_blocks.append(
            StackedConvLayers(60, 60)
        )
        self.encoder_conv_blocks.append(
            StackedConvLayers(60, 120)
        )
        self.encoder_conv_blocks.append(
            StackedConvLayers(120, 120)
        )
        self.encoder_conv_blocks.append(
            StackedConvLayers(120, 240)
        )
        self.encoder_conv_blocks.append(
            StackedConvLayers(240, 240)
        )

        self.encoder_conv_blocks.append(
            StackedConvLayers(240, 320)
        )
        self.encoder_conv_blocks.append(
            StackedConvLayers(320, 240)
        )


        #deconder
        self.decoder_conv_blocks = []
        self.decoder_conv_blocks.append(
            StackedConvLayers(480, 240)
        )
        self.decoder_conv_blocks.append(
            StackedConvLayers(480, 240)
        )
        self.decoder_conv_blocks.append(
            StackedConvLayers(240, 120)
        )
        self.decoder_conv_blocks.append(
            StackedConvLayers(240, 120)
        )
        self.decoder_conv_blocks.append(
            StackedConvLayers(240, 120)
        )
        self.decoder_conv_blocks.append(
            StackedConvLayers(120, 60)
        )
        self.decoder_conv_blocks.append(
            StackedConvLayers(120, 60)
        )
        self.decoder_conv_blocks.append(
            StackedConvLayers(120, 60)
        )
        self.decoder_conv_blocks.append(
            StackedConvLayers(60, 28)
        )
        self.decoder_conv_blocks.append(
            StackedConvLayers(56, 28)
        )
        self.decoder_conv_blocks.append(
            StackedConvLayers(56, 28)
        )
        self.decoder_conv_blocks.append(
            StackedConvLayers(28, 28)
        )

        # output
        self.output = nn.Sequential(
                            nn.Conv3d(28, out_channels-1, kernel_size=3, padding=1),
                            nn.Sigmoid()
                        )

        self.pool = nn.Sequential(
                            nn.MaxPool3d((2, 2, 2))
                        )

        self.up = nn.Sequential(
                            # nn.functional.interpolate(scale_factor=(2, 2, 2), mode='trilinear')
                            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear',align_corners=True)
                        )
        self.encoder_conv_blocks = nn.ModuleList(self.encoder_conv_blocks)
        self.decoder_conv_blocks = nn.ModuleList(self.decoder_conv_blocks)

    def forward(self, x,other_skip=None):

        skips = []
        for i in range(8):
            x = self.encoder_conv_blocks[i](x)
            skips.append(x)
            if (i + 1) % 2 ==0:
                x = self.pool(x)
        x = self.encoder_conv_blocks[8](x)
        x = self.encoder_conv_blocks[9](x)

        for i in range(4):
            x = self.up(x)
            x = torch.cat((x, skips[(7 - i*2)]), dim=1)
            x = self.decoder_conv_blocks[i*3](x)
            x = torch.cat((x, skips[(7 - i*2-1)]), dim=1)
            x = self.decoder_conv_blocks[i*3+1](x)
            x = self.decoder_conv_blocks[i*3+2](x)

        x = self.output(x)
        return x

    def max_stride(self):
        return 16

def vnet_kaiming_init(net):
    net.apply(kaiming_weight_init)


def main():
    input = torch.randn(1,1,32,32,32)
    model = Unet_2skip(1, 3)
    pred= model(input)
    print (11)


if __name__ == "__main__":
    main()