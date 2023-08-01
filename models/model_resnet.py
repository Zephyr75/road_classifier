import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class build_resnet(nn.Module):
    def __init__(self):
        super(build_resnet, self).__init__()

        """ Encoder """
        self.e1 = conv_block(3, 64)
        self.e2 = conv_block(64, 128)
        self.e3 = conv_block(128, 256)
        self.e4 = conv_block(256, 512)
        self.e5 = conv_block(512, 1024)

        """ Bottleneck """
        self.b = conv_block(1024, 2048)

        """ Decoder """
        self.d1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2, padding=0)
        self.d1_2 = conv_block(2048, 1024)
        self.d2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0)
        self.d2_2 = conv_block(1024, 512)
        self.d3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        self.d3_2 = conv_block(512, 256)
        self.d4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.d4_2 = conv_block(256, 128)
        self.d5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.d5_2 = conv_block(128, 64)

        self.d = nn.ModuleList([self.d1, self.d1_2, self.d2, self.d2_2, self.d3, self.d3_2, self.d4, self.d4_2, self.d5, self.d5_2])
        self.e = nn.ModuleList([self.e1, self.e2, self.e3, self.e4, self.e5])
        self.p = nn.MaxPool2d(kernel_size=2, stride=2)
        
        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)


    def forward(self, x):
        skips = []

        for down in self.e:
            x = down(x)
            skips.append(x)
            x = self.p(x)

        x = self.b(x)
        skips = skips[::-1]

        for idx in range(0, len(self.d), 2):
            x = self.d[idx](x)
            skip = skips[idx//2]

            if x.shape != skip.shape:
                x = TF.resize(x, size=skip.shape[2:])

            x = self.d[idx+1](torch.cat((skip, x), dim=1))

        return self.outputs(x)

if __name__ == "__main__":
    x = torch.randn((2, 3, 400, 400))
    f = build_resnet()
    y = f(x)
    print(y.shape)