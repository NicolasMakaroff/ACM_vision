import torch
import torch.nn as nn
from PIL import Image, ImageOps
import numpy as np
import math
import matplotlib.pyplot as plt

import timeit 

class CCV(nn.Module):

    def __init__(self, initial_contours, dt, lambda_, color=False):
        super(CCV, self).__init__()
        self.im, self.c, self.h, self.w = initial_contours.shape
        self.u1 = initial_contours
        self.dt = dt.reshape(-1,1).repeat(1,self.c).reshape(-1,self.c,1,1)
        self.lambda_ = (lambda_ * math.sqrt(torch.pi)/torch.sqrt(dt)).reshape(-1,1).repeat(1,self.c).reshape(-1,self.c,1,1)
        self.color = color

    def __call__(self, input_tensor, maxIter = 100, plot = False):
        im, c, h, w = input_tensor.shape
        P = torch.div(input_tensor, torch.max(torch.max(torch.abs(input_tensor))))
        if self.color:
            for i in range(3):
                P[:,:,:,i] = torch.div(P[:,:,:,i],torch.max(torch.max(torch.abs(P[:,:,:,i]))))
            f1 = P[:,:,:,0] # first color channel
            f2 = P[:,:,:,1] # second
            f3 = P[:,:,:,2] # third
        
        u2 = torch.cuda.FloatTensor(im,c,h,w).fill_(1)- self.u1
        
        NS = 2
        NC = 3 if self.color else 1
    
        delta = 1
        iter = 1

        while iter <= maxIter:
            
            if NS == 2 and NC == 1:
                [f1, f2] = self.daterm(P, self.u1, u2)
                [uh1, uh2] = self.HeatConv( self.u1, u2)

                index1 = f1 + (uh2 * self.lambda_)
                index2 = f2 + (uh1 * self.lambda_)

                u1_af = (index1 <= index2).double()
                u2_af = 1 - u1_af

                delta = torch.sum(torch.abs(u1_af[:,:] - self.u1[:,:]))
                self.u1 = u1_af
                u2 = u2_af
            
            elif NS==2 and NC == 3:
                [f11, f21] = self.daterm(f1, self.u1, u2) # data term of first color channel
                [f12, f22] = self.daterm(f2, self.u1, u2) # data term of second color channel
                [f13, f23] = self.daterm(f3, self.u1, u2) # data term of third color channel
                [uh1,uh2] = self.HeatConv(self.dt, self.u1, u2) # heat kernel convolution
                index1 = f11 + f12 + f13 - (uh1.permute(2,3,0,1) * self.lambda_).permute(2,3,0,1)
                index2 = f21 + f22 + f23 - (uh2.permute(2,3,0,1) * self.lambda_).permute(2,3,0,1)
                u1_af = (index1 <= index2).double() # thresholding if u1>u2 then u1=1 vise versa
                u2_af = 1-u1_af
                delta = torch.sum(torch.abs(u1_af[:,:]-self.u1[:,:]))
                self.u1 = u1_af
                u2 = u2_af

            if plot:
            
                self.plotCurves(input_tensor[1], self.u1[1].numpy(), 'r')
                plt.savefig('img/levelset_%d.png'%(iter), bbox_inches='tight')

            if delta == 0:
                return self.u1
            iter += 1

        return self.u1

    def HeatConv(self, u1, u2):
        im, c, w, h = self.im, self.c, self.h, self.w

        k1 = torch.arange(-w/2, w/2) if w % 2 == 0 else torch.arange(-w/2, w/2)
        k2 = torch.arange(-h/2, h/2) if h % 2 == 0 else torch.arange(-h/2, h/2)

        [K1, K2] = torch.meshgrid(k1, k2, indexing='ij')
        K1, K2 = K1.repeat(im,c, 1, 1).to('cuda:0'), K2.repeat(im,c, 1, 1).to('cuda:0')
        KF = torch.exp((-(torch.square(K1)+torch.square(K2)).permute(2,3,0,1))).permute(2,3,0,1)*self.dt

        K = torch.fft.fftshift(KF,dim=[2,3])

        u_hat1 = torch.real(torch.fft.ifft2(torch.fft.fft2(u1)*K))
        u_hat2 = torch.real(torch.fft.ifft2(torch.fft.fft2(u2)*K))

        return u_hat1, u_hat2

    def daterm(self, f, u1, u2):
        c1 = torch.sum(torch.sum(u1*f))/torch.sum(torch.sum(u1))
        c2 = torch.sum(torch.sum(u2*f))/torch.sum(torch.sum(u2))

        f1 = torch.square(f-c1)  
        f2 = torch.square(f-c2)

        return f1, f2

    def plotCurves(self,image, phi, color):
        plt.ion()
        fig, axes = plt.subplots(ncols=2)
        fig.axes[0].cla()
        fig.axes[0].imshow(image.numpy(),cmap='gray')
        fig.axes[0].contour(phi, 0, colors=color)
        fig.axes[0].set_axis_off()
        plt.draw()

        fig.axes[1].cla()
        fig.axes[1].imshow(phi)
        fig.axes[1].set_axis_off()
        plt.draw()
        
        plt.pause(0.001)


if __name__ == '__main__':
    I1 = Image.open('TCGA_CS_5396_20010302_17.tif').convert('L')
    I2 = Image.open('TCGA_DU_5849_19950405_24.tif').convert('L')

    im1 = ImageOps.autocontrast(I1)
    im2 = ImageOps.autocontrast(I2)
    i1 = np.array(im1)

    i2 = np.array(im2)
    I = np.stack((i1,i2))
    P = torch.FloatTensor(I)
    c, h, w = P.shape

    start = timeit.default_timer()

    u1 = torch.FloatTensor(c, h, w).fill_(0)
    u1[0,50:h-130, 60:w-110] = torch.FloatTensor(h-180, w-170).fill_(1)
    u1[1,30:h-150, 120:w-90] = torch.FloatTensor(h-180, w-210).fill_(1)

    dt = torch.FloatTensor([0.02,0.02])
    lambda_ = torch.FloatTensor([0.05,0.02])

    ccv = CCV(initial_contours = u1, dt = dt, lambda_ = lambda_, color=False)

    ccv(input_tensor = P, maxIter= 10, plot= False)

    end = timeit.default_timer()

    print(end - start)
