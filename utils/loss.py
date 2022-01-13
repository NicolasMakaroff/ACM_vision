import torch.nn as nn
import numpy as np
import torch


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc

eps = np.finfo(float).eps

class ACMLoss(nn.Module):
    
    def __init__(self):
        super(ACMLoss, self).__init__()

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        
        x = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:] # horizontal and vertical directions
        y = y_pred[:,:,:,1:] - y_pred[:,:,:,:-1]
        
        delta_x = x[:,:,1:,:-2]**2
        delta_y = y[:,:,:-2,1:]**2
        delta_u = torch.abs(delta_x + delta_y)

        lenth = torch.mean(torch.sqrt(delta_u + eps)) # equ.(11) in the paper
        
        C_1 = torch.ones((256, 256))
        C_2 = torch.zeros((256, 256))
        C_1 = C_1.type_as(y_pred)  
        C_2 = C_2.type_as(y_pred)  
        region_in = torch.abs(torch.mean( y_pred[:,0,:,:] * ((y_true[:,0,:,:] - C_1)**2) ) ) # equ.(12) in the paper
        region_out = torch.abs(torch.mean( (1-y_pred[:,0,:,:]) * ((y_true[:,0,:,:] - C_2)**2) )) # equ.(12) in the paper

        lambdaP = 1 # lambda parameter could be various.
        mu = 1 # mu parameter could be various.

        return lenth + lambdaP * (mu * region_in + region_out)

