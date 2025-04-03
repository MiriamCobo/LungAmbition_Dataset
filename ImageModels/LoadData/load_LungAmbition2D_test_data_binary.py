import os
import nrrd
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import zoom
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image  # Import PIL for resizing
import torch.nn.functional as F
import torch

# Cropping Strategy in protocaps
# 2D: Crops tightly around the nodule in each slice, excluding surrounding tissue.
# 3D: Crops a region that includes the nodule and may encompass adjacent structures, providing more context.

def calculate_centroid(mask):
    """
    Calculate the centroid of a 3D binary mask.

    Args:
        mask (numpy.ndarray): 3D binary mask array (z, y, x).

    Returns:
        tuple: Centroid coordinates (z, y, x).
    """
    coords = np.argwhere(mask > 0)
    centroid = coords.mean(axis=0)
    return tuple(centroid.astype(int))

def scale_intensity_range(image, a_min, a_max, b_min, b_max, clip=True):
    """Normalize intensity values to a target range."""
    image = (image - a_min) / (a_max - a_min) * (b_max - b_min) + b_min
    if clip:
        image = np.clip(image, b_min, b_max)
    return image

def compute_bounding_box(mask, margin=0):
    """
    Compute the bounding box around the segmentation mask with an optional margin.

    Parameters:
    - mask (np.ndarray): 3D binary mask.
    - margin (int): Extra space to add to the bounding box.

    Returns:
    - min_coords (np.ndarray): Minimum bounding box coordinates.
    - max_coords (np.ndarray): Maximum bounding box coordinates.
    """
    mask_coords = np.array(np.where(mask > 0))  # Get coordinates of the mask
    min_coords = mask_coords.min(axis=1) - margin  # Expand the box
    max_coords = mask_coords.max(axis=1) + 1 + margin  # Include max voxel

    # Ensure bounds stay within image limits
    min_coords = np.maximum(min_coords, 0)
    max_coords = np.minimum(max_coords, np.array(mask.shape))

    return min_coords, max_coords

def compute_bounding_box_2D(mask, margin=0):
    """
    Compute the bounding box around the segmentation mask in 2D (X, Y) with an optional margin.

    Parameters:
    - mask (np.ndarray): 2D binary mask.
    - margin (int): Extra space to add to the bounding box.

    Returns:
    - min_coords (np.ndarray): Minimum bounding box coordinates (y, x).
    - max_coords (np.ndarray): Maximum bounding box coordinates (y, x).
    """
    mask_coords = np.array(np.where(mask > 0))  # Get coordinates of the mask
    min_coords = mask_coords.min(axis=1) - margin  # Expand the box
    max_coords = mask_coords.max(axis=1) + 1 + margin  # Include max voxel

    # Ensure bounds stay within image limits
    min_coords = np.maximum(min_coords, 0)
    max_coords = np.minimum(max_coords, np.array(mask.shape))

    return min_coords, max_coords

def crop_to_nodule_2D(image, mask, centroid, cropout_cube_size=32, margin_too_big_size=0):
    """
    Crops a 2D square region around the nodule in a single slice (X, Y).
    Ensures the output is exactly `cropout_cube_size × cropout_cube_size`.

    Parameters:
    - image (np.ndarray): 2D slice of the scan.
    - mask (np.ndarray): 2D binary mask.
    - centroid (tuple/list/np.ndarray): Center of the nodule.
    - cropout_cube_size (int): Fixed output size.
    - margin_too_big_size (int): Margin to add when the nodule is too large.

    Returns:
    - new_img (np.ndarray): Cropped and resized image slice.
    - new_mask (np.ndarray): Cropped and resized binary mask slice.
    """
    cropout_cube_size_half = cropout_cube_size // 2

    # Ensure mask is not empty
    if np.sum(mask) == 0:
        raise ValueError("The mask is empty. Cannot calculate centroid for an empty mask.")

    # bounding box around nodule
    min_coords, max_coords = compute_bounding_box_2D(mask, margin=margin_too_big_size)
    nodule_mask_dims = max_coords - min_coords

    # Step 2: Initialize crop borders
    cropout_border = np.zeros((2, 2), dtype=int)

    for d in range(2):  # d = 0 (Y), 1 (X)
        if nodule_mask_dims[d] > cropout_cube_size:
            # Use full bounding box along this axis
            cropout_border[d, 0] = min_coords[d]
            cropout_border[d, 1] = max_coords[d]
        else:
            # Fixed-size crop centered at centroid
            lower_bound = int(centroid[d] - cropout_cube_size_half)
            upper_bound = int(centroid[d] + cropout_cube_size_half)

            # Clamp to image bounds
            if lower_bound < 0:
                cropout_border[d, 0] = 0
                cropout_border[d, 1] = cropout_cube_size
            elif upper_bound > image.shape[d]:
                cropout_border[d, 1] = image.shape[d]
                cropout_border[d, 0] = image.shape[d] - cropout_cube_size
            else:
                cropout_border[d, 0] = lower_bound
                cropout_border[d, 1] = upper_bound


    # **Step 2: Crop the image and mask**
    cropped_img = image[cropout_border[0, 0]:cropout_border[0, 1], cropout_border[1, 0]:cropout_border[1, 1]]
    cropped_mask = mask[cropout_border[0, 0]:cropout_border[0, 1], cropout_border[1, 0]:cropout_border[1, 1]]

    # Ensure the final shape is exactly `cropout_cube_size × cropout_cube_size`
    zoom_factors = np.ones(2)  # Default: no zoom
    for d in range(2):
        if cropped_img.shape[d] > cropout_cube_size:
            zoom_factors[d] = cropout_cube_size / cropped_img.shape[d]  # Scale down

    print(f"Zoom factors: {zoom_factors}")

    # Resize image and mask
    new_img = zoom(cropped_img, zoom_factors, order=3)  # Cubic interpolation for image
    new_mask = zoom(cropped_mask, zoom_factors, order=0)  # Nearest-neighbor for mask
    # check new_mask is not empty
    if np.sum(new_mask) == 0:
        raise ValueError("The resulting mask is empty, something happened during calculation.")

    return new_img, new_mask

class LungAmbitionDataset2D(Dataset):
    """
    PyTorch Dataset for 2D LungAmbition data

    Parameters:
    - data_list: List of dictionaries with 'image', 'seg', 'mal', 'id', and 'seg_path'.
    - crop_size: Crop size for y, x dimensions.
    """
    def __init__(self, data_list, crop_size, margin_too_big_size, type_processing="original", augment_prob=0):
        self.data_list = data_list
        self.crop_size = crop_size
        self.margin_too_big_size = margin_too_big_size
        self.type_processing = type_processing
        self.slices = []
        self.augment_prob = augment_prob

        # **Preprocess all slices once to store them**
        for row in self.data_list:
            img_path = row['image']
            seg_path = row['seg']
            identificator = row['id']

            # Read the image and segmentation
            img = sitk.ReadImage(img_path)
            seg = sitk.ReadImage(seg_path)

            # Convert to NumPy arrays
            img_np = sitk.GetArrayFromImage(img)  # (z, y, x)
            seg_np = sitk.GetArrayFromImage(seg)  # (z, y, x)

            # Change order from (z, y, x) to (x, y, z)
            img_np = np.transpose(img_np, (2, 1, 0))
            seg_np = np.transpose(seg_np, (2, 1, 0))

            # Rotate and flip
            img_np = np.rot90(img_np, k=-1, axes=(0, 1))
            seg_np = np.rot90(seg_np, k=-1, axes=(0, 1))
            img_np = np.flip(img_np, axis=1)
            seg_np = np.flip(seg_np, axis=1)

            # Compute lesion size and check crop
            seg_dim_xy = np.array(seg_np.nonzero())[0:2].ptp(axis=1) + 1
            if np.any(seg_dim_xy > self.crop_size):
                print(f"The nodule with id {identificator} is too large for the crop size. Nodule size: {seg_dim_xy}. This nodule will be downsampled")
            
            # Identify the slices where the mask is present
            mask_slices = np.unique(np.where(seg_np > 0)[-1])  # Get unique Z indices

            for z in mask_slices:
                img_slice = img_np[:, :, z]  # Extract the corresponding slice
                mask_slice = seg_np[:, :, z]

                if self.type_processing == "original": ###
                    # Compute bounding box
                    min_coords, max_coords = compute_bounding_box_2D(mask_slice, margin=0)

                    # Crop the image and mask to the bounding box
                    cropped_img = img_slice[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1]]
                    cropped_mask = mask_slice[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1]]

                    # Resize to the desired crop size
                    img_PIL = Image.fromarray(cropped_img.astype(np.float32))
                    mask_PIL = Image.fromarray(cropped_mask.astype(np.uint8))

                    # Resize using PIL
                    img_resized = img_PIL.resize((self.crop_size, self.crop_size), Image.BICUBIC)
                    mask_resized = mask_PIL.resize((self.crop_size, self.crop_size), Image.NEAREST)

                    # Convert back to NumPy
                    new_img = np.asarray(img_resized, dtype=np.float32)
                    new_mask = np.asarray(mask_resized, dtype=np.uint8)
                else:
                    # Compute centroid in (Y, X)
                    centroid = np.array(np.where(mask_slice > 0)).mean(axis=1).astype(int)

                    # Crop and process the 2D slice
                    new_img, new_mask = crop_to_nodule_2D(img_slice, mask_slice, centroid, self.crop_size)

                self.slices.append({
                    "image": new_img,
                    "mask": new_mask,
                    "id": identificator,
                    "malignancy": row["mal"],
                    "seg_path": row["seg_path"],
                    "slice_idx": z  # Store slice index
                })


    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        sample = self.slices[idx]
        img_np = sample["image"]
        mask_np = sample["mask"]

        # Convert to PyTorch tensors
        img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        mask_tensor = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)  # (1, H, W)

        if self.augment_prob > 0 and torch.rand(1).item() < self.augment_prob:
            img_tensor, mask_tensor = self.apply_same_transform_2D(img_tensor, mask_tensor)

        return {
            "image": img_tensor,
            "seg": mask_tensor,
            "id": sample["id"],
            "mal": sample["malignancy"],
            "seg_path": sample["seg_path"],
            "slice_idx": sample["slice_idx"]  # Slice index within the 3D volume
        }
    
    def apply_same_transform_2D(self, image, mask):
        image = image.to(dtype=torch.float64)
        mask = mask.to(dtype=torch.float64)

        angle = torch.empty(1).uniform_(-30, 30).item()
        scale = torch.empty(1).uniform_(0.9, 1.1).item()

        theta = torch.tensor([
            [scale * torch.cos(torch.deg2rad(torch.tensor(angle))), -torch.sin(torch.deg2rad(torch.tensor(angle))), 0],
            [torch.sin(torch.deg2rad(torch.tensor(angle))), scale * torch.cos(torch.deg2rad(torch.tensor(angle))), 0]
        ], dtype=torch.float64).unsqueeze(0)  # [1, 2, 3]

        grid = F.affine_grid(theta, image.shape, align_corners=False)
        transformed_image = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=False)
        transformed_mask = F.grid_sample(mask, grid, mode='nearest', padding_mode='border', align_corners=False)

        return transformed_image.to(dtype=torch.float32), transformed_mask.to(dtype=torch.float32)


def load_lungAmbition_2D(df_merged, batch_size=16, crop_size=32, margin_too_big_size=2, type_processing=None, shuffle=False, augment_prob=0):
    """
    Load and preprocess 2D slices from LungAmbition dataset.
    
    - df_merged: DataFrame with image/mask paths.
    - batch_size: Number of samples per batch.
    - crop_size: Size of initial crop around nodule.
    - shuffle: Whether to shuffle.

    Returns:
    - DataLoader with 2D slices.
    """
    data_list = [] # List of dictionaries with image/mask paths
    for _, row in df_merged.iterrows():
        if len(row['SEG_Files']) > 0:
            img_path = row['NRRD_File']#.replace('tenerife/', '')
            for seg_file in row['SEG_Files']:
                seg_path = seg_file#.replace('tenerife/', '')
                data_list.append({
                    "image": img_path,
                    "seg": seg_path,
                    "mal": row["Malignancy"],
                    "id": row["ID_proteinData"],
                    "seg_path": seg_path
                })
    dataset = LungAmbitionDataset2D(data_list, crop_size=crop_size, margin_too_big_size=margin_too_big_size, type_processing=type_processing, augment_prob=augment_prob)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
