import torch
import blobfile as bf
# from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset

import os
import csv
import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
import random


def load_data_cond(
    *, csv_path, batch_size, crop_size, crop_spacing, deterministic=False
):
    dataset = OccDatasetCond(
       csv_path=csv_path,
        crop_size=crop_size,    
        crop_spacing=crop_spacing, 
        augmentation=True,
        train=True
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def load_nifti(path):
    """Load NIfTI image using SimpleITK and convert to numpy array + spacing + origin"""
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image)  # (D, H, W)
    spacing = image.GetSpacing()           # (x, y, z) in mm
    origin = image.GetOrigin()
    direction = image.GetDirection()
    return array, spacing, origin, direction

def get_center_of_label(mask_np, label=2):
    """Get the centroid of the specified label (physical coordinates)"""
    coords = np.where(mask_np == label)
    if len(coords[0]) == 0:
        return None
    # voxel index
    center_voxel = np.array([np.mean(coords[i]) for i in range(3)])  # [z, y, x]
    return center_voxel

def random_offset(center, max_offset_mm=(10, 10, 10), spacing=(1.0, 1.0, 1.0)):
    
    offset_mm = np.random.uniform(-1, 1, size=3) * max_offset_mm
    offset_voxel = offset_mm / np.array(spacing)  # 转为体素单位
    return center + offset_voxel

def resample_image_to_spacing(image_array, orig_spacing, target_spacing, interpolator=sitk.sitkLinear):
    
    orig_spacing = np.array(orig_spacing)
    target_spacing = np.array(target_spacing)

    if np.allclose(orig_spacing, target_spacing, atol=1e-3):
        return image_array, orig_spacing

    image_sitk = sitk.GetImageFromArray(image_array)
    image_sitk.SetSpacing(orig_spacing[::-1])  # SITK: (x,y,z) → numpy: (z,y,x)

    orig_size = np.array(image_array.shape[::-1])  # (W, H, D)
    new_size = np.round(orig_size * (orig_spacing / target_spacing)).astype(int)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size.tolist())
    resampler.SetOutputSpacing(target_spacing[::-1])
    resampler.SetOutputDirection(image_sitk.GetDirection())
    resampler.SetOutputOrigin(image_sitk.GetOrigin())
    resampler.SetInterpolator(interpolator)
    resampled = resampler.Execute(image_sitk)

    return sitk.GetArrayFromImage(resampled), target_spacing

def crop_3d(image, mask, center_voxel, crop_size_voxels):
    
    cz, cy, cx = center_voxel
    d, h, w = crop_size_voxels

    z0 = int(round(cz - d // 2))
    y0 = int(round(cy - h // 2))
    x0 = int(round(cx - w // 2))

    z1 = z0 + d
    y1 = y0 + h
    x1 = x0 + w

    pad_z0 = max(0, -z0)
    pad_y0 = max(0, -y0)
    pad_x0 = max(0, -x0)
    pad_z1 = max(0, z1 - image.shape[0])
    pad_y1 = max(0, y1 - image.shape[1])
    pad_x1 = max(0, x1 - image.shape[2])

    z0_clip = max(0, z0)
    y0_clip = max(0, y0)
    x0_clip = max(0, x0)
    z1_clip = min(image.shape[0], z1)
    y1_clip = min(image.shape[1], y1)
    x1_clip = min(image.shape[2], x1)

    cropped_img = image[z0_clip:z1_clip, y0_clip:y1_clip, x0_clip:x1_clip]
    cropped_mask = mask[z0_clip:z1_clip, y0_clip:y1_clip, x0_clip:x1_clip]

    cropped_img = np.pad(cropped_img, ((pad_z0, pad_z1), (pad_y0, pad_y1), (pad_x0, pad_x1)), mode='constant', constant_values=-1000)
    cropped_mask = np.pad(cropped_mask, ((pad_z0, pad_z1), (pad_y0, pad_y1), (pad_x0, pad_x1)), mode='constant', constant_values=0)

    return cropped_img, cropped_mask

class OccDatasetCond(Dataset):
    def __init__(
        self,
        csv_path,
        crop_size,      
        crop_spacing,   
        augmentation=True,
        max_offset_mm=(10, 10, 10),
        flip_prob=0.5,
        scale_range=(0.9, 1.1),         
        train=True
    ):
        super().__init__()
        self.csv_path = csv_path
        df = pd.read_csv(csv_path)
        self.image_paths = df['image_path'].tolist()
        self.mask_paths = df['mask_path'].tolist()
        self.labels = df['label'].tolist()
        assert len(self.image_paths) == len(self.mask_paths)
        
        self.crop_size = np.array(crop_size)  # (H, W, D) in mm
        self.base_spacing = np.array(crop_spacing)  # (x, y, z) — SITK order
        self.augmentation = augmentation
        self.max_offset_mm = np.array(max_offset_mm)
        self.flip_prob = flip_prob
        self.scale_range = scale_range
        self.train = train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        label = self.labels[idx]

        img_np, img_spacing, _, _ = load_nifti(img_path)      
        mask_np, mask_spacing, _, _ = load_nifti(mask_path)   # (D, H, W)

        assert np.allclose(img_spacing, mask_spacing, atol=1e-2), f"Spacing mismatch: {img_path}"

        original_spacing = np.array(img_spacing)  # (x, y, z)
        
        center_voxel = get_center_of_label(mask_np, label=1)
        if center_voxel is None:
            
            center_voxel = get_center_of_label(mask_np, label=1)
            if center_voxel is None:
                center_voxel = np.array([
                    np.random.randint(0, img_np.shape[0]),
                    np.random.randint(0, img_np.shape[1]),
                    np.random.randint(0, img_np.shape[2])
                ])

        if self.augmentation and self.train:
            scale_factor = np.random.uniform(*self.scale_range)
            aug_spacing = original_spacing * scale_factor
        else:
            aug_spacing = original_spacing

        img_resampled, _ = resample_image_to_spacing(
            img_np, original_spacing, aug_spacing, interpolator=sitk.sitkLinear
        )
        mask_resampled, _ = resample_image_to_spacing(
            mask_np, original_spacing, aug_spacing, interpolator=sitk.sitkNearestNeighbor
        )

        physical_center = center_voxel[::-1] * original_spacing  # (x, y, z)
        new_center_voxel = physical_center / aug_spacing
        new_center_voxel = new_center_voxel[::-1]  # 转回 (z, y, x)

        if self.augmentation and self.train:
            new_center_voxel = random_offset(
                new_center_voxel,
                max_offset_mm=self.max_offset_mm,
                spacing=aug_spacing
            )


        crop_size_voxels = self.crop_size

        cropped_img, cropped_mask = crop_3d(
            img_resampled, mask_resampled, new_center_voxel, crop_size_voxels
        )

        if self.augmentation and self.train:
            for axis, prob in enumerate([self.flip_prob, self.flip_prob, self.flip_prob]):
                if random.random() < prob:
                    cropped_img = np.flip(cropped_img, axis=axis).copy()
                    cropped_mask = np.flip(cropped_mask, axis=axis).copy()

        window_center, window_width = 300, 400
        cropped_img = (cropped_img - window_center) / window_width 
        cropped_img = np.clip(cropped_img, -1, 1)     

        cropped_img = np.transpose(cropped_img, (2,1,0))
        cropped_mask = np.transpose(cropped_mask, (2,1,0))

        masked_cropped_img = np.copy(cropped_img)
        masked_cropped_img[cropped_mask==1] = 0
        masked_cropped_img[cropped_mask>0] = 1

        image_tensor = torch.from_numpy(cropped_img).float().unsqueeze(0)  # (1, D, H, W)
        image_mask_tensor = torch.from_numpy(masked_cropped_img).float().unsqueeze(0)  # (1, D, H, W)
        mask_tensor = torch.from_numpy(cropped_mask.astype(np.int64)).long().unsqueeze(0)    
        
        input_tensor = torch.concat([image_tensor, image_mask_tensor, mask_tensor], dim=0)
        
        label = torch.tensor(label, dtype=torch.long)
        out_dict = {'y':label}
        # print(input_tensor.shape)

        return input_tensor, out_dict