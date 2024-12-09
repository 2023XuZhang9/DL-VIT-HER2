import os
import cv2
import numpy as np
import SimpleITK as sitk
import pandas as pd
import torch

def create_soft_mask(hard_mask, blur_kernel_size=21, blur_sigma=10):
    # Convert hard mask to soft mask
    soft_mask = cv2.GaussianBlur(hard_mask.astype(float), (blur_kernel_size, blur_kernel_size), blur_sigma)
    # Ensure mask values are between 0 and 1
    soft_mask = np.clip(soft_mask, 0, 1)
    return soft_mask

def weighted_blending(image, soft_mask, alpha=0.3):
    # Ensure image and soft_mask are both float type for precise calculations
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    if soft_mask.dtype != np.float32:
        soft_mask = soft_mask.astype(np.float32)

    # Normalize soft_mask to ensure its range is between 0 and 1
    soft_mask = cv2.normalize(soft_mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Compute weighted blended image
    enhanced_image = image * soft_mask  # Enhance the image using the mask
    blended_image = cv2.addWeighted(image, alpha, enhanced_image, 1-alpha, 0)
    return blended_image

def gen_dataset(data_dir, id_list, clini_df, img_size=224, outline=10, dataset_type="train"):
    # Define a function named gen_dataset that takes in data directory, image ID list, clinical data clini_df,
    # and optional parameters: output image size img_size and outline for region of interest
    data_dir = data_dir 
    # Set the directory path
    data_for_excel = pd.DataFrame(columns=['ID', 'HER2_subtype'])
    img_tensor_list = []
    HER2_subtype_list = []

    # Create CLAHE object outside the loop to reuse it
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Initialize lists to store processed images and related clinical data
    for id in id_list:
        # Iterate through each ID in id_list
        # Read NIfTI file and find the slice index with the largest ROI area
        ROI = sitk.ReadImage(os.path.join(data_dir, str(id), 'label.nii'))
        ROI_data = sitk.GetArrayFromImage(ROI).astype(np.int16)
        ROI_data = np.transpose(ROI_data, [1, 2, 0])
        focus_index = np.argmax(np.sum(np.sum(ROI_data, axis=0), axis=0))
        # Read the corresponding slice
        nii_file = sitk.ReadImage(os.path.join(data_dir, str(id), "005.nii"))
        X = sitk.GetArrayFromImage(nii_file).astype(np.float32)
        X = X[focus_index, :, :]
        m = ROI_data[:, :, focus_index]

        # Load the original image (X) and mask image (m) for each ID from the corresponding directory
        x, y = np.where(m > 0)
        # Find all non-zero pixels in the mask image
        w0, h0 = m.shape
        x_min = max(0, int(np.min(x) - outline))
        x_max = min(w0, int(np.max(x) + outline))
        y_min = max(0, int(np.min(y) - outline))
        y_max = min(h0, int(np.max(y) + outline))
        # Calculate the boundaries of the region of interest, adjust based on the outline parameter
        m = m[x_min:x_max, y_min:y_max]
        X = X[x_min:x_max, y_min:y_max]
        # Create soft mask
        soft_m = create_soft_mask(m, blur_kernel_size=21, blur_sigma=10)
        # Crop the mask and original image to keep only the region of interest
        X = np.uint8(cv2.normalize(X, None, 0, 255, cv2.NORM_MINMAX)) # Normalize the original image to range 0-255 and convert to uint8
        X = clahe.apply(X)  # Apply CLAHE enhancement
        X = X.astype(np.float32)
        # Weighted blending of the original image with the soft mask
        X = weighted_blending(X, m, alpha=0.3)
        X = np.uint8(X) # Convert back to uint8 if needed
        # Apply Z-Score normalization
        X = (X - np.mean(X)) / np.std(X)
        # X = (X - np.min(X)) / (np.max(X) - np.min(X))
        X_m_1 = X.copy()
        # Create a copy of the original image
        if X_m_1.shape[0] != img_size or X_m_1.shape[1] != img_size:
            X_m_1 = cv2.resize(X_m_1, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        # Resize the image if its size is not equal to img_size

        X_m_1 = np.expand_dims(X_m_1, axis=-1)
        X_m_1 = np.concatenate([X_m_1, X_m_1, X_m_1], axis=-1)
        # Expand the single channel image to three channels by copying the single channel three times

        X_m_1 = X_m_1.transpose((2, 0, 1))  # Move the channel dimension to the front
        img_tensor = torch.tensor(X_m_1).float()
        img_tensor_list.append(img_tensor)

        # Extract clinical data
        id = clini_df[clini_df.id == id]['id'].values[0]
        HER2_subtype_value = clini_df[clini_df.id == id]['HER2_subtype'].values[0]

        data_for_excel = pd.concat([data_for_excel, pd.DataFrame([{
            'ID': id, 
            'HER2_subtype': HER2_subtype_value  
        }])], ignore_index=True)
        # Save DataFrame to Excel file
        excel_filename = f'clinical_data_{dataset_type}.xlsx'
        data_for_excel.to_excel(excel_filename, index=False)
        
        HER2_subtype_list.append(clini_df[clini_df.id == id]['HER2_subtype'].values[0])

    return img_tensor_list, HER2_subtype_list
