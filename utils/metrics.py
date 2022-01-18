import torch
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as edt



class HausdorffDistance:
    def hd_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:

        if np.count_nonzero(x) == 0 or np.count_nonzero(y) == 0:
            return np.array([np.Inf])

        indexes = np.nonzero(x)
        distances = edt(np.logical_not(y))

        return np.array(np.max(distances[indexes]))

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert (
            pred.shape[1] == 1 and target.shape[1] == 1
        ), "Only binary channel supported"

        pred = (pred > 0.5).byte()
        target = (target > 0.5).byte()

        right_hd = torch.from_numpy(
            self.hd_distance(pred.cpu().numpy(), target.cpu().numpy())
        ).float()

        left_hd = torch.from_numpy(
            self.hd_distance(target.cpu().numpy(), pred.cpu().numpy())
        ).float()

        return torch.max(right_hd, left_hd)


def precision(x, y):
    n = torch.sum(y)
    print(x)
    p_ = torch.sum(x)
    return p_ / n

def recall(x, y):
    pass
    
def iou_coef(y_true, y_pred, smooth=1):
  intersection = torch.sum(torch.abs(y_true * y_pred))
  union = torch.sum(y_true,[1,2,3])+torch.sum(y_pred,[1,2,3])-intersection
  iou = torch.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou


def compute_pre_rec(gt,mask,mybins=np.arange(0,256)/255):
    #print(gt.shape)
    #print(mask.shape)
    if(len(gt.shape)<2 or len(mask.shape)<2):
        print("ERROR: gt or mask is not matrix!")
        exit()
    """if(len(gt.shape)>2): # convert to one channel
        gt = gt[:,:,0]
    if(len(mask.shape)>2): # convert to one channel
        mask = mask[:,:,0]"""
    if(gt.shape!=mask.shape):
        print("ERROR: The shapes of gt and mask are different!")
        exit()
    #print(np.max(gt))
    #print(np.max(mask))
    #print(np.min(mask))
    precision_sum = 0
    recall_sum = 0
    for i in range(gt.shape[0]):
        x = gt[i]
        #print(x.shape)
        y = mask[i]
        gtNum = x[x>0.5] # pixel number of ground truth foreground regionsi
        #print(gtNum.shape)
        gtNum = gtNum.size
        pp = y[x>0.5] # mask predicted pixel values in the ground truth foreground region
        #print(pp.shape)
        nn = y[x<=0.5] # mask predicted pixel values in the ground truth background region
        #print(nn.shape)
        
        true_positive = np.sum(pp > 0.5)
        false_negative = np.sum(pp <= 0.5)
        true_negative = np.sum(nn <= 0.5)
        false_positive = np.sum(nn > 0.5)

        precision_sum += true_positive / (true_positive + false_positive + 1e-12)
        recall_sum += true_positive / (true_positive + false_negative + 1e-12)
        
    precision = precision_sum / gt.shape[0]
    recall = recall_sum / gt.shape[0]
    f1 = 2 * (precision * recall)/(precision + recall)
    return precision, recall, f1
