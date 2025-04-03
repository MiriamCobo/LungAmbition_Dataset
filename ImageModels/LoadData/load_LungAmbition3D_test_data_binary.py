import os
import nrrd
import numpy as np
import pandas as pd
from tqdm import trange
import SimpleITK as sitk
import h5py
from scipy.ndimage import zoom
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import random
# to read this folder do from ..loadLungAmbitionData.load_LungAmbition3D_test_data import load_lungAmbition

def calculate_centroid(mask):
    """
    Calculate the centroid of a 3D binary mask.

    Args:
        mask (numpy.ndarray): 3D binary mask array (z, y, x).

    Returns:
        tuple: Centroid coordinates (z, y, x).
    """
    # Get indices of non-zero elements
    coords = np.argwhere(mask > 0)
    
    # Calculate the mean of each dimension
    centroid = np.round(coords.mean(axis=0)).astype(int)
    
    return tuple(centroid)

def scale_intensity_range(image, a_min, a_max, b_min, b_max, clip=True):
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

def compute_cube_size(mask, margin_too_big_size=0):
    """
    Computes the full bounding box dimensions of a 3D mask and ensures the cube includes the full nodule
    by adding the margin_too_big_size.

    Parameters:
    - mask (np.ndarray): 3D binary mask.
    - margin_too_big_size (int): Margin to expand the bounding box.

    Returns:
    - cube_size (np.ndarray): Bounding box size along (Z, Y, X).
    - min_coords (np.ndarray): Minimum coordinates of the bounding box.
    - max_coords (np.ndarray): Maximum coordinates of the bounding box.
    """
    mask_coords = np.array(mask.nonzero())  # Get all nonzero voxel coordinates

    if mask_coords.shape[1] == 0:  # If the mask is empty
        raise ValueError("The mask is empty. Cannot compute bounding box size.")

    # Compute bounding box min/max
    min_coords = mask_coords.min(axis=1) - margin_too_big_size
    max_coords = mask_coords.max(axis=1) + 1 + margin_too_big_size  # +1 to include the last voxel

    # Ensure bounds stay within image limits
    min_coords = np.maximum(min_coords, 0)
    max_coords = np.minimum(max_coords, np.array(mask.shape))

    # Compute final cube size
    cube_size = max_coords - min_coords

    return cube_size

def crop_to_nodule(image, mask, centroid, cropout_cube_size=48, margin_too_big_size=0):
    """
    Crops a cube around the nodule from the given image and mask.

    Parameters:
    - image (np.ndarray): 3D volume of the scan.
    - mask (np.ndarray): 3D binary mask of the nodule.
    - centroid (tuple/list/np.ndarray): Center of the nodule.
    - cropout_cube_size (int): Size of the cubic crop. Default is 48.
    - margin_too_big_size (int): Margin to add to the bounding box when the nodule is too large.

    Returns:
    - new_img (np.ndarray): Cropped image volume.
    - new_mask (np.ndarray): Cropped binary mask.
    """
    cropout_cube_size_half = cropout_cube_size // 2

    # Ensure mask is not empty
    if np.sum(mask) == 0:
        raise ValueError("The mask is empty. Cannot calculate centroid for an empty mask.")

    # Compute the size of the cube including the whole mask (not only the nodule!)
    nodule_mask_dims = compute_cube_size(mask)

    if np.any(nodule_mask_dims > cropout_cube_size):
        # **Nodule is too large in at least one dimension → Apply selective zoom**
        print(f"The nodule is too large for the crop size. Mask volume size: {nodule_mask_dims}")

        min_coords, max_coords = compute_bounding_box(mask, margin_too_big_size)

        # **Step 1: Crop a `cropout_cube_size` region normally in valid dimensions**
        cropout_border = np.zeros((3, 2), dtype=int)

        for d in range(3):
            if nodule_mask_dims[d] > cropout_cube_size:
                # If dimension is too large, take full bounding box
                cropout_border[d, 0] = min_coords[d]
                cropout_border[d, 1] = max_coords[d]
            else:
                # Crop a fixed `cropout_cube_size` around the centroid
                lower_bound = int(centroid[d] - cropout_cube_size_half)
                upper_bound = int(centroid[d] + cropout_cube_size_half)

                # Ensure crop stays within image bounds
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
        cropped_img = image[cropout_border[0, 0]:cropout_border[0, 1],
                            cropout_border[1, 0]:cropout_border[1, 1],
                            cropout_border[2, 0]:cropout_border[2, 1]]

        cropped_mask = mask[cropout_border[0, 0]:cropout_border[0, 1],
                            cropout_border[1, 0]:cropout_border[1, 1],
                            cropout_border[2, 0]:cropout_border[2, 1]]

        # **Step 3: Resize only dimensions that exceed cropout_cube_size**
        crop_shape = np.array(cropped_img.shape)
        zoom_factors = np.ones(3)  # Default: no zoom in any axis
        for d in range(3):
            if crop_shape[d] > cropout_cube_size:
                zoom_factors[d] = cropout_cube_size / crop_shape[d]  # Scale down only this axis

        print(f"Selective zoom factors: {zoom_factors}")

        # **Resize image and mask**
        new_img = zoom(cropped_img, zoom_factors, order=3)  # Cubic interpolation for image
        new_mask = zoom(cropped_mask, zoom_factors, order=0)  # Nearest-neighbor interpolation for mask

    else:
        # Define cropout borders using centroid
        cropout_border = np.zeros((3, 2), dtype=int)

        for d in range(3):
            lower_bound = int(centroid[d] - cropout_cube_size_half)
            upper_bound = int(centroid[d] + cropout_cube_size_half)

            # Ensure crop stays within image bounds
            if lower_bound < 0:
                cropout_border[d, 0] = 0
                cropout_border[d, 1] = cropout_cube_size
            elif upper_bound > image.shape[d]:
                cropout_border[d, 1] = image.shape[d]
                cropout_border[d, 0] = image.shape[d] - cropout_cube_size
            else:
                cropout_border[d, 0] = lower_bound
                cropout_border[d, 1] = upper_bound
        # Crop the image and mask
        new_img = image[cropout_border[0, 0]:cropout_border[0, 1],
                        cropout_border[1, 0]:cropout_border[1, 1],
                        cropout_border[2, 0]:cropout_border[2, 1]]

        new_mask = mask[cropout_border[0, 0]:cropout_border[0, 1],
                        cropout_border[1, 0]:cropout_border[1, 1],
                        cropout_border[2, 0]:cropout_border[2, 1]]


    # Ensure the cropped mask is not empty
    if new_mask.sum() == 0:
        raise ValueError("The mask is empty. Check!")
    # check output sizes are cropout_cube_size, cropout_cube_size, cropout_cube_size
    assert new_img.shape == (cropout_cube_size, cropout_cube_size, cropout_cube_size), f"Image shape is {new_img.shape}"

    return new_img, new_mask

class LungAmbitionDataset(Dataset):
    """
    PyTorch Dataset for the LungAmbition project.

    Parameters:
    - data_list: List of dictionaries with 'image', 'seg', 'mal', 'id', and 'seg_path'.
    - crop_size: Size of the cropped cube.
    """
    def __init__(self, data_list, crop_size, type_processing="original", augment_prob=0):
        self.data_list = data_list
        self.crop_size = crop_size
        self.type_processing = type_processing

        self.augment_prob = augment_prob  # Probability of applying augmentations

    def __len__(self):
        return len(self.data_list)
    
    def apply_same_transform(self, image, mask):
        image = image.to(dtype=torch.float64)
        mask = mask.to(dtype=torch.float64)
        # Generate random parameters
        angle = torch.empty(1).uniform_(-30, 30).item()  # Rotation
        scale = torch.empty(1).uniform_(0.9, 1.1).item()  # Scale between 0.9x and 1.1x

        # Convert degrees to radians for affine grid
        theta = torch.tensor([
            [scale * torch.cos(torch.deg2rad(torch.tensor(angle))), -torch.sin(torch.deg2rad(torch.tensor(angle))), 0],
            [torch.sin(torch.deg2rad(torch.tensor(angle))), scale * torch.cos(torch.deg2rad(torch.tensor(angle))), 0]
        ], dtype=torch.float64).unsqueeze(0)  # Shape: [1, 2, 3]

        # Generate affine grid for transformation
        grid = F.affine_grid(theta, image.shape, align_corners=False)

        # Apply the same transformation to both image and mask
        transformed_image = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=False)
        transformed_mask = F.grid_sample(mask, grid, mode='nearest', padding_mode='border', align_corners=False)

        # Convert back to required data types
        transformed_image = transformed_image.to(dtype=torch.float64)  # Keep image as float64
        transformed_mask = transformed_mask.to(dtype=torch.int32)  # Convert mask back to int32
        
        return transformed_image, transformed_mask

    def __getitem__(self, idx):
        row = self.data_list[idx]
        img_path = row['image']
        seg_path = row['seg']
        identificator = row['id']

        # Read the image and segmentation
        img = sitk.ReadImage(img_path)
        seg = sitk.ReadImage(seg_path)

        # Convert to NumPy arrays and preprocess
        img_np = sitk.GetArrayFromImage(img)
        seg_np = sitk.GetArrayFromImage(seg)

        # Change order from (z, y, x) to (x, y, z)
        img_np = np.transpose(img_np, (2, 1, 0))
        seg_np = np.transpose(seg_np, (2, 1, 0))

        # Rotate and flip
        img_np = np.rot90(img_np, k=-1, axes=(0, 1))
        seg_np = np.rot90(seg_np, k=-1, axes=(0, 1))
        img_np = np.flip(img_np, axis=1)
        seg_np = np.flip(seg_np, axis=1)

        if self.type_processing == "original": ###
            centroid = np.array(calculate_centroid(seg_np))  # Get (x, y, z) centroid

            # **Step 2: Define crop size and borders**
            cropout_cube_size = self.crop_size  # 48³ Cube
            cropout_cube_size_half = cropout_cube_size // 2
            cropout_border = np.array([[0, img_np.shape[0]], 
                                        [0, img_np.shape[1]], 
                                        [0, img_np.shape[2]]])  # Default to full image bounds

            # **Step 3: Adjust borders based on centroid position**
            for d in range(3):  # Iterate over (x, y, z) dimensions
                lower_bound = int(centroid[d] - cropout_cube_size_half)
                upper_bound = int(centroid[d] + cropout_cube_size_half)

                if lower_bound < 0:  # If out of bounds, shift to [0:crop_size]
                    cropout_border[d, 1] = cropout_cube_size
                elif upper_bound > img_np.shape[d]:  # If exceeding image size, shift
                    cropout_border[d, 0] = img_np.shape[d] - cropout_cube_size
                else:  # Normal case, keep it centered around centroid
                    cropout_border[d, 0] = lower_bound
                    cropout_border[d, 1] = upper_bound

            # **Step 4: Crop image and mask using computed borders**
            new_img = img_np[cropout_border[0, 0]:cropout_border[0, 1],
                             cropout_border[1, 0]:cropout_border[1, 1],
                             cropout_border[2, 0]:cropout_border[2, 1]]

            new_mask = seg_np[cropout_border[0, 0]:cropout_border[0, 1],
                              cropout_border[1, 0]:cropout_border[1, 1],
                              cropout_border[2, 0]:cropout_border[2, 1]]

            # **Check size consistency**
            assert new_img.shape == (self.crop_size, self.crop_size, self.crop_size), f"Image shape is {new_img.shape}, expected {self.crop_size}"
            assert new_mask.shape == (self.crop_size, self.crop_size, self.crop_size), f"Mask shape is {new_mask.shape}, expected {self.crop_size}"
        else:
            # Check nodule size and handle cropping/downsampling
            seg_dim = np.array(seg_np.nonzero()).ptp(axis=1) + 1
            if np.any(seg_dim > self.crop_size):
                print(f"The nodule with id {identificator} is too large for the crop size. Nodule size: {seg_dim}. This nodule will be downsampled")

            # Calculate centroid and preprocess
            centroid = np.array(calculate_centroid(seg_np)).astype(int)
            vol = scale_intensity_range(img_np, -1000, 400, 0, 1, clip=True)
            new_img, new_mask = crop_to_nodule(vol, seg_np, centroid, self.crop_size)

        # Convert to PyTorch tensors
        new_img = torch.tensor(new_img, dtype=torch.float64).unsqueeze(0)  # Add channel dimension
        new_mask = torch.tensor(new_mask, dtype=torch.int32).unsqueeze(0)  # Add channel dimension

        # Apply Augmentation
        if self.augment_prob > 0 and torch.rand(1).item() < self.augment_prob:
            new_img, new_mask = self.apply_same_transform(new_img, new_mask)
        
        return {
            "image": new_img,
            "seg": new_mask,
            "id": identificator,
            "mal": row["mal"],
            "seg_path": row["seg_path"]
        }

def load_lungAmbition(df_merged, batch_size, spatial_size=[48, 48, 48], shuffle=False, type_processing=None, augment_prob=0):
    """
    Load and preprocess data into a PyTorch DataLoader.

    Parameters:
    - df_merged: DataFrame containing metadata about images and segmentations.
    - batch_size: Batch size for the DataLoader.
    - spatial_size: Desired spatial size of the cropped cubes (default is [48, 48, 48]).
    - shuffle: Whether to shuffle the data (default is False).

    Returns:
    - DataLoader: PyTorch DataLoader with preprocessed data.
    """
    data_list = []
    assert spatial_size[0] == spatial_size[1] == spatial_size[2], "Spatial size must be cubic."
    crop_size = spatial_size[0]

    # Prepare the dataset metadata
    for _, row in df_merged.iterrows():
        if len(row['SEG_Files']) > 0:
            img_path = row['NRRD_File']#.replace('/home/ubuntu/tenerife/', '/gpfs/projects/computacion/cobom/')
            for seg_file in row['SEG_Files']:
                seg_path = seg_file#.replace('/home/ubuntu/tenerife/', '/gpfs/projects/computacion/cobom/')
                data_list.append({
                    "image": img_path,
                    "seg": seg_path,
                    "mal": row["Malignancy"],
                    "id": row["ID_proteinData"],
                    "seg_path": seg_path
                })
    # Create the Dataset and DataLoader
    dataset = LungAmbitionDataset(data_list, crop_size, type_processing, augment_prob=augment_prob)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1,
        pin_memory=torch.cuda.is_available()
    )

    return loader