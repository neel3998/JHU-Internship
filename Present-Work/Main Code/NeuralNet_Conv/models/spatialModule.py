from torch import nn
import pytorch_spiking
import torch
from torch.nn.modules.activation import LeakyReLU, Sigmoid
from models.config import FNNconfigs

'''
The entire data can be represented using:

1. Type of curve.
2. Max amplitude.
3. x,y value.
4. Velocity of palpation.
'''
class EncConvBlock(nn.Module):
    def __init__(self):
        super(EncConvBlock, self).__init__()
        FNNconfigs.__init__(self,)

        hiddenCH = [32, 16 , 4, 1]
        convModule = []
        convModule.append(nn.Sequential(nn.Conv2d(3,hiddenCH[0], 3,2,1),
                                        nn.BatchNorm2d(hiddenCH[0]),
                                        nn.LeakyReLU()
                                        )
                                    )
        for i in range(0,len(hiddenCH)-1):
            convModule.append(
                nn.Sequential(
                    nn.Conv2d(hiddenCH[i],hiddenCH[i+1], 3, 2, 1),
                    nn.BatchNorm2d(hiddenCH[i+1]),
                    nn.LeakyReLU()
                )
            )

        self.model = nn.Sequential(*convModule)
        self.fc21 = nn.Linear(8,4)
        self.fc22 = nn.Linear(8,4)

    def forward(self, x):

        a = self.model(x)
        a = torch.reshape(a, (-1,8))
        mean = self.fc21(a)
        var = self.fc21(a)

        return mean, var

def reparametrize(mu, logvar):
    std = torch.exp(logvar * 0.5)
    eps = torch.rand_like(std)
    ret = eps * std + mu
    return ret

class DecConvBlock(nn.Module):
    def __init__(self):
        super(DecConvBlock, self).__init__()
        FNNconfigs.__init__(self,)

        hiddenCH = [32, 16 , 4, 1]
        hiddenCH.reverse()
        deconvModule = []

        for i in range(0,len(hiddenCH)-1):
            deconvModule.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hiddenCH[i],hiddenCH[i+1], 3, 2, 1, 1),
                    nn.BatchNorm2d(hiddenCH[i+1]),
                    nn.LeakyReLU()
                )
            )
        deconvModule.append(nn.Sequential(nn.ConvTranspose2d(hiddenCH[-1], 3 , 3, 2,1, 1),
                                        nn.BatchNorm2d(3),
                                        # nn.LeakyReLU()
                                        nn.Sigmoid()
                                        )
                                    )
        self.model = nn.Sequential(*deconvModule)
        self.fc = nn.Linear(4,8)


    def forward(self, x):

        # a = self.model(x)
        a = self.fc(x)
        a = torch.reshape(a, (-1,1,1,8))


        return self.model(a)

class ZDecBlock(nn.Module):
    def __init__(self):
        super(ZDecBlock, self).__init__()
        hidden_dims = [8,16, 64]

        trajModule = []
        trajModule.append(
            nn.Sequential(
                nn.Linear(4,hidden_dims[0]),
                nn.LeakyReLU()
        ))
        for i in range(len(hidden_dims)-1):
            trajModule.append(
                nn.Sequential(
                nn.Linear(hidden_dims[i],hidden_dims[i+1]),
                nn.LeakyReLU()
                )   
            )
        
        trajModule.append((
            nn.Sequential(
                nn.Linear(hidden_dims[-1],16),
                nn.Sigmoid()
            )
        ))

        self.model = nn.Sequential(*trajModule)
        
    
    def forward(self, x):
        z = self.model(x)

        return z

class smallVAEencBlock(nn.Module):
    def __init__(self):
        super(smallVAEencBlock, self).__init__()
        FNNconfigs.__init__(self,)

        self.model = nn.Sequential(
                    nn.Conv2d(3,1, 3, 2, 1),
                    nn.BatchNorm2d(1),
                    nn.LeakyReLU()
                )

        self.fc21 = nn.Linear(4,4)
        self.fc22 = nn.Linear(4,4)

    def forward(self, x):

        a = self.model(x)
        a = torch.reshape(a, (-1,4))
        mean = self.fc21(a)
        var = self.fc21(a)

        return mean, var

def reparametrize(mu, logvar):
    std = torch.exp(logvar * 0.5)
    eps = torch.rand_like(std)
    ret = eps * std + mu
    return ret