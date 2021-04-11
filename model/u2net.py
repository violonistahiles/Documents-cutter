import torch
import torch.nn as nn


class REBNCONVp(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONVp, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x_in = x
        x_out = self.relu_s1(self.bn_s1(self.conv_s1(x_in)))

        return x_out


def _upsample_like(src, tar):
    size = tar.shape[2:]
    src = nn.functional.interpolate(src, size, mode='bilinear', align_corners=False)
    return src


class RSU7(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        # Encoder
        self.rebnconvin = REBNCONVp(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONVp(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONVp(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONVp(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONVp(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONVp(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONVp(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONVp(mid_ch, mid_ch, dirate=2)

        # Decoder
        self.rebnconv6d = REBNCONVp(2 * mid_ch, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONVp(2 * mid_ch, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONVp(2 * mid_ch, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONVp(2 * mid_ch, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONVp(2 * mid_ch, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONVp(2 * mid_ch, out_ch, dirate=1)

    def forward(self, x):
        x_in = x

        x_in = self.rebnconvin(x_in)

        x_1 = self.rebnconv1(x_in)
        x = self.pool1(x_1)

        x_2 = self.rebnconv2(x)
        x = self.pool2(x_2)

        x_3 = self.rebnconv3(x)
        x = self.pool3(x_3)

        x_4 = self.rebnconv4(x)
        x = self.pool4(x_4)

        x_5 = self.rebnconv5(x)
        x = self.pool5(x_5)

        x_6 = self.rebnconv6(x)

        x_7 = self.rebnconv7(x_6)

        x_6d = self.rebnconv6d(torch.cat((x_7, x_6), 1))
        x_6_up = _upsample_like(x_6d, x_5)

        x_5d = self.rebnconv5d(torch.cat((x_6_up, x_5), 1))
        x_5_up = _upsample_like(x_5d, x_4)

        x_4d = self.rebnconv4d(torch.cat((x_5_up, x_4), 1))
        x_4_up = _upsample_like(x_4d, x_3)

        x_3d = self.rebnconv3d(torch.cat((x_4_up, x_3), 1))
        x_3_up = _upsample_like(x_3d, x_2)

        x_2d = self.rebnconv2d(torch.cat((x_3_up, x_2), 1))
        x_2_up = _upsample_like(x_2d, x_1)

        x_1d = self.rebnconv1d(torch.cat((x_2_up, x_1), 1))

        return x_1d + x_in


class RSU6(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        # Encoder
        self.rebnconvin = REBNCONVp(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONVp(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONVp(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONVp(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONVp(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONVp(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNCONVp(mid_ch, mid_ch, dirate=2)

        # Decoder
        self.rebnconv5d = REBNCONVp(2 * mid_ch, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONVp(2 * mid_ch, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONVp(2 * mid_ch, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONVp(2 * mid_ch, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONVp(2 * mid_ch, out_ch, dirate=1)

    def forward(self, x):
        x_in = x

        x_in = self.rebnconvin(x_in)

        x_1 = self.rebnconv1(x_in)
        x = self.pool1(x_1)

        x_2 = self.rebnconv2(x)
        x = self.pool2(x_2)

        x_3 = self.rebnconv3(x)
        x = self.pool3(x_3)

        x_4 = self.rebnconv4(x)
        x = self.pool4(x_4)

        x_5 = self.rebnconv5(x)

        x_6 = self.rebnconv6(x_5)

        x_5d = self.rebnconv5d(torch.cat((x_6, x_5), 1))
        x_5_up = _upsample_like(x_5d, x_4)

        x_4d = self.rebnconv4d(torch.cat((x_5_up, x_4), 1))
        x_4_up = _upsample_like(x_4d, x_3)

        x_3d = self.rebnconv3d(torch.cat((x_4_up, x_3), 1))
        x_3_up = _upsample_like(x_3d, x_2)

        x_2d = self.rebnconv2d(torch.cat((x_3_up, x_2), 1))
        x_2_up = _upsample_like(x_2d, x_1)

        x_1d = self.rebnconv1d(torch.cat((x_2_up, x_1), 1))

        return x_1d + x_in


class RSU5(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()

        # Encoder
        self.rebnconvin = REBNCONVp(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONVp(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONVp(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONVp(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONVp(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONVp(mid_ch, mid_ch, dirate=2)

        # Decoder
        self.rebnconv4d = REBNCONVp(2 * mid_ch, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONVp(2 * mid_ch, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONVp(2 * mid_ch, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONVp(2 * mid_ch, out_ch, dirate=1)

    def forward(self, x):
        x_in = x

        x_in = self.rebnconvin(x_in)

        x_1 = self.rebnconv1(x_in)
        x = self.pool1(x_1)

        x_2 = self.rebnconv2(x)
        x = self.pool2(x_2)

        x_3 = self.rebnconv3(x)
        x = self.pool3(x_3)

        x_4 = self.rebnconv4(x)

        x_5 = self.rebnconv5(x_4)

        x_4d = self.rebnconv4d(torch.cat((x_5, x_4), 1))
        x_4_up = _upsample_like(x_4d, x_3)

        x_3d = self.rebnconv3d(torch.cat((x_4_up, x_3), 1))
        x_3_up = _upsample_like(x_3d, x_2)

        x_2d = self.rebnconv2d(torch.cat((x_3_up, x_2), 1))
        x_2_up = _upsample_like(x_2d, x_1)

        x_1d = self.rebnconv1d(torch.cat((x_2_up, x_1), 1))

        return x_1d + x_in


class RSU4(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        # Encoder
        self.rebnconvin = REBNCONVp(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONVp(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONVp(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONVp(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONVp(mid_ch, mid_ch, dirate=2)

        # Decoder
        self.rebnconv3d = REBNCONVp(2 * mid_ch, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONVp(2 * mid_ch, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONVp(2 * mid_ch, out_ch, dirate=1)

    def forward(self, x):
        x_in = x

        x_in = self.rebnconvin(x_in)

        x_1 = self.rebnconv1(x_in)
        x = self.pool1(x_1)

        x_2 = self.rebnconv2(x)
        x = self.pool2(x_2)

        x_3 = self.rebnconv3(x)

        x_4 = self.rebnconv4(x_3)

        x_3d = self.rebnconv3d(torch.cat((x_4, x_3), 1))
        x_3_up = _upsample_like(x_3d, x_2)

        x_2d = self.rebnconv2d(torch.cat((x_3_up, x_2), 1))
        x_2_up = _upsample_like(x_2d, x_1)

        x_1d = self.rebnconv1d(torch.cat((x_2_up, x_1), 1))

        return x_1d + x_in


class RSU4F(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        # Encoder
        self.rebnconvin = REBNCONVp(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONVp(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONVp(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONVp(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = REBNCONVp(mid_ch, mid_ch, dirate=8)

        # Decoder
        self.rebnconv3d = REBNCONVp(2 * mid_ch, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONVp(2 * mid_ch, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONVp(2 * mid_ch, out_ch, dirate=1)

    def forward(self, x):
        x_in = x
        x_in = self.rebnconvin(x_in)

        x_1 = self.rebnconv1(x_in)
        x_2 = self.rebnconv2(x_1)
        x_3 = self.rebnconv3(x_2)

        x_4 = self.rebnconv4(x_3)

        x_3d = self.rebnconv3d(torch.cat((x_4, x_3), 1))
        x_2d = self.rebnconv2d(torch.cat((x_3d, x_2), 1))
        x_1d = self.rebnconv1d(torch.cat((x_2d, x_1), 1))

        return x_1d + x_in


class U2NET(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()

        # Encoder
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # Decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6, out_ch, 1)

    def forward(self, x):
        # Encoder
        x_in = x

        x_1 = self.stage1(x_in)
        x = self.pool12(x_1)

        x_2 = self.stage2(x)
        x = self.pool23(x_2)

        x_3 = self.stage3(x)
        x = self.pool34(x_3)

        x_4 = self.stage4(x)
        x = self.pool45(x_4)

        x_5 = self.stage5(x)
        x = self.pool56(x_5)

        x_6 = self.stage6(x)
        x_6_up = _upsample_like(x_6, x_5)

        # Decoder
        x_5d = self.stage5d(torch.cat((x_6_up, x_5), 1))
        x_5_up = _upsample_like(x_5d, x_4)

        x_4d = self.stage4d(torch.cat((x_5_up, x_4), 1))
        x_4_up = _upsample_like(x_4d, x_3)

        x_3d = self.stage3d(torch.cat((x_4_up, x_3), 1))
        x_3_up = _upsample_like(x_3d, x_2)

        x_2d = self.stage2d(torch.cat((x_3_up, x_2), 1))
        x_2_up = _upsample_like(x_2d, x_1)

        x_1d = self.stage1d(torch.cat((x_2_up, x_1), 1))

        # output
        d1 = self.side1(x_1d)

        d2 = self.side2(x_2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(x_3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(x_4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(x_5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(x_6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(
            d4), torch.sigmoid(d5), torch.sigmoid(d6)
