import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import PIL
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self, conv_dim):
        super(ResidualBlock, self).__init__()
        """
        Residual blocks help the model to effectively learn the transformation from one domain to another. 
        """
        self.conv1 = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1,
                          instance_norm=True)
        self.conv2 = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1,
                          instance_norm=True)

    def forward(self, x):
        out_1 = F.relu(self.conv1(x))
        out_2 = x + self.conv2(out_1)
        return out_2


def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=False, instance_norm=False,
           dropout=False, dropout_ratio=0.5):
    """
    Creates a transpose convolutional layer, with optional batch / instance normalization. Select either batch OR instance normalization.
    """

    # Add layers
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))

    # Batch normalization
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    # Instance normalization
    if instance_norm:
        layers.append(nn.InstanceNorm2d(out_channels))

    # Dropout
    if dropout:
        layers.append(nn.Dropout2d(dropout_ratio))

    return nn.Sequential(*layers)


class CycleGenerator(nn.Module):

    def __init__(self, conv_dim=64, n_res_blocks=6):
        super(CycleGenerator, self).__init__()
        """
        Input is RGB image (256x256x3) while output is a single value

        determine size = [(W−K+2P)/S]+1
        W: input=256
        K: kernel_size=4
        P: padding=1
        S: stride=2
        """

        # Encoder layers
        self.conv1 = conv(in_channels=3, out_channels=conv_dim, kernel_size=4)  # (128, 128, 64)
        self.conv2 = conv(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=4,
                          instance_norm=True)  # (64, 64, 128)
        self.conv3 = conv(in_channels=conv_dim * 2, out_channels=conv_dim * 4, kernel_size=4,
                          instance_norm=True)  # (32, 32, 256)

        # Residual blocks (number depends on input parameter)
        res_layers = []
        for layer in range(n_res_blocks):
            res_layers.append(ResidualBlock(conv_dim * 4))
        self.res_blocks = nn.Sequential(*res_layers)

        # Decoder layers
        self.deconv4 = deconv(in_channels=conv_dim * 4, out_channels=conv_dim * 2, kernel_size=4,
                              instance_norm=True)  # (64, 64, 128)
        self.deconv5 = deconv(in_channels=conv_dim * 2, out_channels=conv_dim, kernel_size=4,
                              instance_norm=True)  # (128, 128, 64)
        self.deconv6 = deconv(in_channels=conv_dim, out_channels=3, kernel_size=4, instance_norm=True)  # (256, 256, 3)

    def forward(self, x):
        """
        Given an image x, returns a transformed image.
        """

        # Encoder
        out = F.leaky_relu(self.conv1(x), negative_slope=0.2)  # (128, 128, 64)
        out = F.leaky_relu(self.conv2(out), negative_slope=0.2)  # (64, 64, 128)
        out = F.leaky_relu(self.conv3(out), negative_slope=0.2)  # (32, 32, 256)

        # Residual blocks
        out = self.res_blocks(out)

        # Decoder
        out = F.leaky_relu(self.deconv4(out), negative_slope=0.2)  # (64, 64, 128)
        out = F.leaky_relu(self.deconv5(out), negative_slope=0.2)  # (128, 128, 64)
        out = torch.tanh(self.deconv6(out))  # (256, 256, 3)

        return out


transform = transforms.ToPILImage()
device = torch.device('cpu')
convert_tensor = transforms.ToTensor()

def resize_in(input_img) :
    input1 = Image.open(input_img)
    input2 = input1.resize((256,256))
    input3 = convert_tensor(input2)
    return input3


def resize_out(output_img) :
    save_image(output_img, 'str.png') # 저장 경로
    img = Image.open('str.png')
    output = img.resize((1080,1920))
    return output

def Gan_prc(image) :
    GAN = torch.load('Gustave2_G_XtoY.pt', map_location=torch.device('cpu'))
    GAN.eval()
    image2 = resize_in(image)
    fake_Y = GAN(image2.to(device))
    result = resize_out(fake_Y)
    return fake_Y

# sample ="F:/Final project/data/CNN/train/Piña colada (cocktail)162.jpg"
# sample = "./이미지1.jpeg"
#
# Gan_prc(sample)


if __name__ == '__main__':

    Gan_prc("./이미지1.jpeg")

