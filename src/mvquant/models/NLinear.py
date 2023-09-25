import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NLinear(nn.Module):
    """
    Normalization-Linear
    https://arxiv.org/pdf/2205.13504.pdf
    """
    def __init__(self, configs):
        super(NLinear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.enc_in
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x: [Batch, Input length, Channel]
        seq_last = x_enc[:,-1:,:].detach()
        x_enc = x_enc - seq_last
        if self.individual:
            output = torch.zeros([x_enc.size(0),self.pred_len,x_enc.size(2)],dtype=x_enc.dtype).to(x_enc.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x_enc[:,:,i])
            x_enc = output
        else:
            x_enc = self.Linear(x_enc.permute(0,2,1)).permute(0,2,1)
        x_enc = x_enc + seq_last
        return x_enc # [Batch, Output length, Channel]
    