import os
import random

import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset

from utils.utils import crop_sample, pad_sample, resize_sample, normalize_volume
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
num_cores = multiprocessing.cpu_count()


class BrainSegmentationDataset(Dataset):
    """Brain MRI dataset for FLAIR abnormality segmentation"""

    in_channels = 3
    out_channels = 1

    def __init__(self, images_dir,transform=None, image_size=256, subset="train", random_sampling=False, validation_cases=10, seed=42):
        
        assert subset in ["all", "train", "validation"]

        # read images
        volumes = {}
        masks = {}
        bbox = {}
        print("reading {} images...".format(subset))
        for (dirpath, dirnames, filenames) in os.walk(images_dir):
            image_slices = []
            mask_slices = []
            bbox_slices = []
            for filename in sorted(filter(lambda f: ".tif" in f, filenames), key=lambda x: int(x.split(".")[-2].split("_")[4])):
                filepath = os.path.join(dirpath, filename)
                if "mask" in filename:
                    img = imread(filepath, as_gray=True)
                    mask_slices.append(img)
                    pos = np.where(img)
                    #bbox_slices.append([np.min[pos[1]], np.max[pos[1]], np.min[pos[0]], np.max[pos[0]]])
                else:
                    image_slices.append(imread(filepath))
            if len(image_slices) > 0:
                patient_id = dirpath.split("/")[-1]
                volumes[patient_id] = np.array(image_slices[1:-1])
                masks[patient_id] = np.array(mask_slices[1:-1])
                #bbox[patient_id] = np.array(bbox_slices[1:-1])

        self.patients = sorted(volumes)

        # select cases to subset
        if not subset == "all":
            random.seed(seed)
            validation_patients = random.sample(self.patients, k=validation_cases)
            if subset == "validation":
                self.patients = validation_patients
            else:
                self.patients = sorted(list(set(self.patients).difference(validation_patients)))

        print("preprocessing {} volumes...".format(subset))
        # create list of tuples (volume, mask)
        patients = tqdm(self.patients)
        self.volumes = [(volumes[k], masks[k]) for k in patients]

        print("cropping {} volumes...".format(subset))
        # crop to smallest enclosing volume
        volumes = tqdm(self.volumes)
        self.volumes = [crop_sample(v) for v in volumes]

        print("padding {} volumes...".format(subset))
        volumes = tqdm(self.volumes)
        # pad to square
        self.volumes = [pad_sample(v) for v in volumes]

        print("resizing {} volumes...".format(subset))
        # resize
        #def Loop(i):
         #   return resize_sample(i, size=image_size)
        #timer = tqdm(self.volumes, desc="resizing")
        #self.volumes = Parallel(n_jobs=20,verbose=1)(delayed(Loop)(i) for i in timer)
        
        volumes = tqdm(self.volumes)
        self.volumes = [resize_sample(v, size=image_size) for v in volumes]

        print("normalizing {} volumes...".format(subset))
        volumes = tqdm(self.volumes)
        # normalize channel-wise
        self.volumes = [(normalize_volume(v), m) for v, m in volumes]

        # probabilities for sampling slices based on masks
        self.slice_weights = [m.sum(axis=-1).sum(axis=-1) for v, m in self.volumes]
        self.slice_weights = [
            (s + (s.sum() * 0.1 / len(s))) / (s.sum() * 1.1) for s in self.slice_weights
        ]

        # add channel dimension to masks
        self.volumes = [(v, m[..., np.newaxis]) for (v, m) in self.volumes]

        print("done creating {} dataset".format(subset))

        # create global index for patient and slice (idx -> (p_idx, s_idx))
        num_slices = [v.shape[0] for v, m in self.volumes]
        self.patient_slice_index = list(
            zip(
                sum([[i] * num_slices[i] for i in range(len(num_slices))], []),
                sum([list(range(x)) for x in num_slices], []),
            )
        )

        self.random_sampling = random_sampling

        self.transform = transform

    def __len__(self):
        return len(self.patient_slice_index)

    def __getitem__(self, idx):
        patient = self.patient_slice_index[idx][0]
        slice_n = self.patient_slice_index[idx][1]

        if self.random_sampling:
            patient = np.random.randint(len(self.volumes))
            slice_n = np.random.choice(
                range(self.volumes[patient][0].shape[0]), p=self.slice_weights[patient]
            )

        v, m = self.volumes[patient]
        image = v[slice_n]
        mask = m[slice_n]

        if self.transform is not None:
            image, mask = self.transform((image, mask))

        # fix dimensions (C, H, W)
        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)
        pos = np.where(mask)
        if pos[0].size == 0:
            bbox = np.array([0,0,0,0])
        else:
            xmin = np.min(pos[1]) - 10 if np.min(pos[1]) >= 10 else 0 
            xmax = np.max(pos[1]) + 10 if (mask.shape[1] - np.max(pos[1])) >= 10 else mask.shape[1]
            ymin = np.min(pos[0]) - 10 if np.min(pos[0]) >= 10 else 0 
            ymax = np.max(pos[0]) + 10 if (mask.shape[1] - np.max(pos[0])) >= 10 else mask.shape[1]
            bbox = np.array([xmin, xmax, ymin, ymax])

        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))
        bbox_tensor = torch.from_numpy(bbox.astype(np.float32))
        # return tensors
        return image_tensor, mask_tensor, bbox_tensor
