from torch import nn
import torch
import numpy as np

class timeModule(nn.Module):
    def __init__(self):
        super(timeModule, self).__init__()
        self.lstm = nn.LSTM(3,3,1, batch_first = True)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1,128,3).cuda(), # hidden state
                torch.zeros(1,128,3).cuda()) # cell state
    
    def forward(self,x):
        x = x.permute(0,2,1)
        # exit()

        output = torch.zeros_like(x)
        for i in range(x.shape[0]):
            inputs = x[i].view(len(x[i]),1,-1)
            hidden = self.init_hidden()
            out, hidden = self.lstm(inputs, hidden)
            output[i] = out.squeeze(1)

        output = output.permute(0,2,1)
