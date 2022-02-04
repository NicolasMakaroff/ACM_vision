from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
from utils.metrics import compute_pre_rec, HausdorffDistance
import wandb
import pytorch_lightning as pl
from utils.loss import ACMLoss, DiceLoss
from utils.utils import  gray2rgb, wb_mask, image2np
from utils.ccv import CCV
from torchvision.models.feature_extraction import create_feature_extractor
from torch import Tensor

acm_loss = ACMLoss()
dsc_loss = DiceLoss()
hd_loss = HausdorffDistance()

class UNet(pl.LightningModule):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        
        #acc = DiceLoss()
        #self.train_acc = acc.clone()
        #self.valid_acc = acc.clone()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.encoder11 = UNet._block(features, features, name="enc11")
        self.encoder12 = UNet._block(features, features, name="enc12")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")
        self.fc1 = nn.Linear(512*4*4,1000)
        self.fc2 = nn.Linear(1000,128)
        self.fc3 = nn.Linear(128,32)
        self.fc4 = nn.Linear(32,2)
        self.relu = nn.ReLU()

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.acmattention4 = ACMAttention(256)
        self.acmattention3 = ACMAttention(128)
        self.acmattention2 = ACMAttention(64)
        self.acmattention1 = ACMAttention(32)
        
    def forward_retro(self, x):
        enc1 = self.encoder1(x)
        enc11 = self.encoder11(enc1)
        enc12 = self.encoder12(enc11)
        enc2 = self.encoder2(self.pool1(enc12))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))
        x = bottleneck.view(-1,512*4*4)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        x = bottleneck.view(-1, 512*4*4)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))

        center = self.upconv4(bottleneck)
        dec4 = torch.cat((center, enc4), dim=1)
        alp1 = self.acmattention4(enc4, enc4, dec4, dt=x[:,0], lambda_=x[:,1])
        dec4 = self.decoder4(alp1.float() * dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        alp2 = self.acmattention3(enc3, enc3, dec3, dt=x[:,0], lambda_=x[:,1])
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        alp3 = self.acmattention2(enc2, enc2, dec2, dt=x[:,0], lambda_=x[:,1])
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        alp4 = self.acmattention1(enc1, enc1, dec1, dt=x[:,0], lambda_=x[:,1])
        dec1 = self.decoder1(dec1)
        
        return torch.sigmoid(self.conv(dec1))


    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    
    def training_step(self, batch, batch_idx):

        x, y_true = batch
        y_pred = self(x)
        #loss = acm_loss(y_pred, y_true)
        loss = dsc_loss(y_pred,y_true)
        #self.log('loss', loss_info, prog_bar = False, on_step=False,on_epoch=True,logger=True)

        return {'loss':loss} #{'loss':loss, 'preds': y_preds, 'targets': y_true}
    
    def training_step_end(self, outs):
        # log accuracy on each step_end, for compatibility with data-parallel
        #self.train_acc(outs["preds"], outs["targets"])
        self.log("train/acc_step", outs)
 
    def training_epoch_end(self, outs):
        outs = outs[0]
        # additional log mean accuracy at the end of the epoch
        self.log("train/acc_epoch", outs['loss'])

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        #feature_extractor = create_feature_extractor(self, return_nodes=['encoder12.enc12relu2',])
        #out = feature_extractor(x)['encoder12.enc12relu2']
        #print(type(res))
        #print(res.shape)
        #print(out.shape)
        #for i in range(out.shape[0]):
            #wandb.log({'layer4':wandb.Image(out[4][i].detach())})
        #val_loss = acm_loss(y_hat, y)
        dsc = dsc_loss(y_hat,y)
        hdd = hd_loss.compute(y_hat, y).item()
        precision, recall, specificity, f1 = compute_pre_rec(y.cpu().numpy(), y_hat.cpu().numpy())
        #self.log('val_loss', val_loss, prog_bar=False, on_step=False, on_epoch=True, logger=True)
        #self.log('val_dsc',dsc, prog_bar=False, on_step=False,on_epoch=True, logger=True)
        #self.log('val_haussdorf',hdd, prog_bar=False, on_step=False,on_epoch=True, logger=True)
        #self.log('precision',precision, prog_bar=False, on_step=False,on_epoch=True, logger=True)
        #self.log('recall', recall, prog_bar=False, on_step=False,on_epoch=True, logger=True)
        #self.log('F1', f1, prog_bar=False, on_step=False,on_epoch=True, logger=True)
        #self.log('specificity', specificity, prog_bar=False, on_step=False,on_epoch=True, logger=True)

        return {'dice': dsc, 'hausdorff': hdd, 'precision': precision, 'recall': recall, 'specificity': specificity, 'F1': f1} 

    def validation_step_end(self, outs):
        self.log('test/dice', outs['dice'], prog_bar=False, on_step=False,on_epoch=True, logger=True)
        self.log('test/hausdorff', outs['hausdorff'], prog_bar=False, on_step=False, on_epoch=True, logger=True)
        self.log('test/precision', outs['precision'], prog_bar=False, on_step=False, on_epoch=True, logger=True)
        self.log('test/recall', outs['recall'], prog_bar=False, on_step=False, on_epoch=True, logger=True)
        self.log('test/specificity', outs['specificity'], prog_bar=False, on_step=False, on_epoch=True, logger=True)
        self.log('test/F1', outs['F1'], prog_bar=False, on_step=False, on_epoch=True, logger=True)

    def validation_epoch_end(self, outs):
        outs = outs[0]
        self.log('test/dice', outs['dice'], prog_bar=False, on_step=False,on_epoch=True, logger=True)
        self.log('test/hausdorff', outs['hausdorff'], prog_bar=False, on_step=False, on_epoch=True, logger=True)
        self.log('test/precision', outs['precision'], prog_bar=False, on_step=False, on_epoch=True, logger=True)
        self.log('test/recall', outs['recall'], prog_bar=False, on_step=False, on_epoch=True, logger=True)
        self.log('test/specificity', outs['specificity'], prog_bar=False, on_step=False, on_epoch=True, logger=True)
        self.log('test/F1', outs['F1'], prog_bar=False, on_step=False, on_epoch=True, logger=True)


class ImagePredictionLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=10):
        super().__init__()
        self.val_imgs, self.val_labels = val_samples
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_labels = self.val_labels[:num_samples]
          
    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)

        logits = pl_module(val_imgs)
        #feature_extractor = create_feature_extractor(self, return_nodes=['decoder2.dec2relu2',])
        #out = feature_extractor(x)['decoder2.dec2relu2'][:,10,:,:]
        mask_list = []
        for original_image, logits, ground_truth in zip(val_imgs, logits, self.val_labels):
            # the raw background image as a numpy array
            #bg_image = image2np(original_image.data)
            bg_image = gray2rgb(original_image[1].cpu().numpy()).astype(np.uint8)
            # run the model on that image
            #prediction = pl_module(original_image)[0]
            prediction_mask = image2np(logits.data).astype(np.uint8)
            # ground truth mask
            true_mask = image2np(ground_truth.data).astype(np.uint8)
            # keep a list of composite images
            mask_list.append(wb_mask(bg_image, prediction_mask, true_mask))

        # log all composite images to W&B
        wandb.log({"predictions" : mask_list})


class ACMAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.g_conv = nn.Conv2d(channels, 512, kernel_size=1, bias=False)
        self.x_conv = nn.Conv2d(channels*2, 512, kernel_size=1, bias=False)
        self.contour_conv = nn.Conv2d(channels, 512, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.channels = channels
        
    def forward(self, contour: Tensor, g: Tensor, x: Tensor, dt: Tensor, lambda_: Tensor) -> Tensor:
        z = self.relu(self.g_conv(g)) + self.x_conv(x)
        print(z.shape)
        print(contour.shape)
        contour = self.contour_conv(contour)
        contour = nn.functional.interpolate(contour, size=(z.shape[2],z.shape[3]),mode='bicubic')
        ccv = CCV(initial_contours = contour, dt=dt, lambda_=lambda_, color=False)
        attention = ccv(input_tensor=z, maxIter=10, plot=False)
        
        alpha = self.activation(attention)
        return alpha
         
