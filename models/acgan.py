import torch
import torch.nn as nn

image_size = 64
batch_size = 32
label_dim = 6
nc = 1
nz = 100

class Generator(nn.Module):
    r""" It is mainly based on the mobile net network as the backbone network generator.
    Args:
        image_size (int): The size of the image. (Default: 64)
        channels (int): The channels of the image. (Default: 1)
        num_classes (int): Number of classes for dataset. (Default: 6)
    """
    def __init__(self, image_size: int = 64, channels: int = 1, num_classes: int = 6) -> None:
        super(Generator, self).__init__()
        self.image_size = image_size
        self.channels = channels
        self.num_classes = num_classes
        self.label_emb = nn.Linear(label_dim, nz)

        # self.l1 = nn.Linear(nz, nz + nz)

        self.main = nn.Sequential(
            # 1*1*200 ->4*4*512 
            nn.ConvTranspose2d(nz + nz, 64*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64*8),
            nn.LeakyReLU(0.2, inplace=True),

            # 4*4*512 -> 8*8*256
            nn.ConvTranspose2d(64*8, 64*4, 4, 2, 1, bias=True),
            nn.Dropout(0.4, inplace=True),
            nn.BatchNorm2d(64*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8*8*256 -> 16*16*128
            nn.ConvTranspose2d(64*4, 64*2, 4, 2, 1, bias=True),
            nn.Dropout(0.4, inplace=True),
            nn.BatchNorm2d(64*2),
            nn.LeakyReLU(0.2, inplace=True),

            # 16*16*128 -> 32*32*64
            nn.ConvTranspose2d(64*2, 64, 4, 2, 1, bias=True),
            nn.Dropout(0.4, inplace=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 32*32*64 -> 64*64*1
            nn.ConvTranspose2d( 64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x, y):
        y = self.label_emb(y)
        
        # gen_input = torch.mul(y, x)
        # gen_input = x + y
        y=y.reshape(-1,nz,1,1)
        x=x.reshape(-1,nz,1,1)
        out = torch.cat([x, y] , dim=1)
        # out = self.l1(gen_input)
        out = out.view(-1, nz+nz, 1, 1)
        out = self.main(out)
        
        return out


# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, nc=1):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            nn.Conv2d(nc, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64 * 2, 4, 2, 1),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64*2 , 64*4, 4, 2, 1),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64*4 , 64*8, 4, 2, 1),
            # nn.Dropout(0.5, inplace=True),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True)
            
            #nn.Conv2d(64*8, 1, 4, 1, 0, bias=False),
            #nn.Dropout(0.4, inplace=True),
            #nn.Sigmoid()
        )
        # The height and width of downsampled image
        ds_size = image_size // 2 ** 4

        self.adv_layer = nn.Sequential(
            nn.Linear(64 * 8 * ds_size ** 2, 64 * 8), 
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64 * 8, 64), 
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid())
        self.aux_layer = nn.Sequential(
            nn.Linear(64 * 8 * ds_size ** 2, 64 * 8), 
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64 * 8, 64), 
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, label_dim),
            nn.Sigmoid())
        
    def forward(self, img):
        out = self.main(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label