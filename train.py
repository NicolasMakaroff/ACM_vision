import argparse
import os

import wandb 

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.dataset import BrainSegmentationDataset as Dataset
from utils.loss import DiceLoss, ACMLoss
from utils.transforms import transforms
from nn.unet import ImagePredictionLogger, UNet

import pytorch_lightning
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
import matplotlib.pyplot as plt
import time

from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

def data_loaders(args,seed=1):
    dataset_train, dataset_valid = datasets(args,seed)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        #num_workers=args.workers,
        #worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        drop_last=False,
        #num_workers=args.workers,
        #worker_init_fn=worker_init,
    )

    return loader_train, loader_valid


def datasets(args,seed=1):
    train = Dataset(
        images_dir=args.images,
        subset="train",
        image_size=args.image_size,
        transform=transforms(scale=args.aug_scale, angle=args.aug_angle, flip_prob=0.5),
    )
    valid = Dataset(
        images_dir=args.images,
        subset="validation",
        image_size=args.image_size,
        random_sampling=True,
        seed=1
    )
    return train, valid

def makedirs(args):
    os.makedirs(args.weights, exist_ok=True)

def main(args,seed):
    makedirs(args)

    #wandb.login()
    #wandb.init(group='ddp')
    #wandb_logger = WandbLogger(project="UNet_pytorch",group='ddp')
    trainloader, valoader = data_loaders(args,seed)
    model = UNet()
    #nodes, _ = get_graph_node_names(model)
    #feature_extractor = create_feature_extractor(model, return_nodes=['encoder1.enc1conv1', 'encoder1.enc1norm1', 'encoder1.enc1relu1',])
    #print(feature_extractor(torch.zeros(1,3,64,64)))
    #print(nodes)
    
    samples = next(iter(valoader))
    trainer = Trainer(
        gpus=[0],
        num_nodes=1,
        accelerator='ddp',
        #logger = wandb_logger,
        #progress_bar_refresh_rate=0,
        max_epochs=args.epochs,
        #benchmark=True,
        check_val_every_n_epoch=1,
        #callbacks=[ImagePredictionLogger(samples)]
    )

    trainer.fit(model,trainloader,valoader)
    #wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser( description="Training U-Net model for segmentation of brain MRI")
    parser.add_argument("--batch-size", type=int, default=16, help="input batch size for training (default: 16)")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train (default: 100)")
    parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate (default: 0.001)")
    parser.add_argument("--device", type=str, default="cuda:0", help="device for training (default: cuda:0)")
    parser.add_argument("--workers",type=int,default=1, help="number of workers for data loading (default: 4)")
    parser.add_argument("--weights", type=str, default="./weights", help="folder to save weights")
    parser.add_argument("--images", type=str, default="./data/kaggle_3m", help="root folder with images")
    parser.add_argument("--image-size",type=int,default=64,help="target input image size (default: 256)")
    parser.add_argument("--aug-scale",type=int,default=0.05,help="scale factor range for augmentation (default: 0.05)")
    parser.add_argument("--aug-angle",type=int,default=15,help="rotation angle range in degrees for augmentation (default: 15)")
    args = parser.parse_args()


    main(args,42)

