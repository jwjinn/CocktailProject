
from ganClass import cover
from ganClass import ResidualBlock
from ganClass import CycleGenerator


if __name__ == '__main__':

    k = cover()

    k.imageLocation('/home/joo/images/gan/이미지1.jpeg')

    k.saveLocation('/home/joo/images/gan/output/str.png')

    k.Gan_prc()

