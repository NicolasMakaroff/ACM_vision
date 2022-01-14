import torch.nn as nn
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


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

class HDDTBinaryLoss(nn.Module):
    def __init__(self):
        """
        compute Hausdorff loss for binary segmentation
        https://arxiv.org/pdf/1904.10030v1.pdf        
        """
        super(HDDTBinaryLoss, self).__init__()


    def forward(self, net_output, target):
        """
        net_output: (batch_size, 2, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        """
        net_output = softmax_helper(net_output)
        pc = net_output[:, 1, ...].type(torch.float32)
        gt = target[:,0, ...].type(torch.float32)
        with torch.no_grad():
            pc_dist = compute_edts_forhdloss(pc.cpu().numpy()>0.5)
            gt_dist = compute_edts_forhdloss(gt.cpu().numpy()>0.5)
        # print('pc_dist.shape: ', pc_dist.shape)
        
        pred_error = (gt - pc)**2
        dist = pc_dist**2 + gt_dist**2 # \alpha=2 in eq(8)

        dist = torch.from_numpy(dist)
        if dist.device != pred_error.device:
            dist = dist.to(pred_error.device).type(torch.float32)

        multipled = torch.einsum("bxyz,bxyz->bxyz", pred_error, dist)
        hd_loss = multipled.mean()

        return hd_loss


def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

def compute_edts_forhdloss(segmentation):
    res = np.zeros(segmentation.shape)
    for i in range(segmentation.shape[0]):
        posmask = segmentation[i]
        negmask = ~posmask
        res[i] = distance_transform_edt(posmask) + distance_transform_edt(negmask)
    return res