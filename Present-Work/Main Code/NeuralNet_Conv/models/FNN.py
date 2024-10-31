from datetime import time
from typing_extensions import runtime
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models.spatialModule import EncConvBlock,DecConvBlock ,reparametrize, ZDecBlock
from models.temporalModule import timeModule
from models.config import FNNconfigs
torch.autograd.set_detect_anomaly(True)

def image_like_reshape(data):
    a=data[0].transpose(1,0)
    for i in range(1,16):
        a = torch.vstack((a, data[i].transpose(1,0)))
    
    a = torch.reshape(a, (128, 16, 3))
    a = a.permute(2,1,0)
    cubic_data = a
    return cubic_data

class FingerNeuralNet(pl.LightningModule):

    def __init__(self):
        super(FingerNeuralNet, self).__init__()
        FNNconfigs.__init__(self,)

        self.learning_rate = 0.2290867652767775
        # self.isSpiking = 0
        # self.timeModule = timeModule()
        self.timeModule = []
        for i in range(16):
            self.timeModule.append(timeModule().cuda())
        # self.convMod = nn.Conv1d(3,1,3,padding=1)
        self.encMod = EncConvBlock()
        self.decMod = DecConvBlock()
        self.Zdec = ZDecBlock()



    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        z = self.timeModule(x)
        z = self.lin(z)
        return z    

    def run_epoch(self,batch):
        time_data = []
        z_data = batch['z0']
        for i in range(16):
            time_data.append(self.timeModule[i](batch['v'+str(i)].float()))
            if i>0:
                z_data = torch.hstack((z_data, batch['z'+str(i)]))
        
        
        time_data = torch.stack(time_data)
        cubic_data = []
        for i in range(10):
            cubic_data.append(image_like_reshape(time_data[:,i]))

        cubic_data = torch.stack(cubic_data)


        mean, var = self.encMod(cubic_data)
        sample = reparametrize(mean, var)
        dec = self.decMod(sample)

        z = self.Zdec(sample)

        reconMSE = torch.abs(F.mse_loss(dec, cubic_data.detach(), reduction='mean'))
        kld = -0.5 * torch.mean(1 + var - torch.pow(mean, 2) - torch.exp(var)).float()
        zMSE = torch.abs(F.mse_loss(z, z_data.float(), reduction='mean'))

        loss = kld + reconMSE + zMSE

        return loss

    def training_step(self, batch, hiddens):
        # self.timeModule.hidden = self.timeModule.init_hidden()
        loss = self.run_epoch(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, 
                               prog_bar=True, logger= True)
        return loss

    def validation_step(self, batch, hiddens):
        loss = self.run_epoch(batch)        
        # Logging to TensorBoard by default
        self.log('val_loss', loss,  on_step=True, on_epoch=True, 
                               prog_bar=True, logger= True)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer