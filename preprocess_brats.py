import nibabel as nib
import numpy as np
import os
import imageio

input_dir = 'MICCAI_BraTS2020_TrainingData'
output_image_dir = 'BraTS_slices/images'
output_mask_dir = 'BraTS_slices/masks'

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

def normalize(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
    return (image * 255).astype(np.uint8)

for folder in os.listdir(input_dir):
    case_path = os.path.join(input_dir, folder)
    flair_path = os.path.join(case_path, f"{folder}_flair.nii")  # 🔧 여기
    mask_path  = os.path.join(case_path, f"{folder}_seg.nii")    # 🔧 여기

    if not os.path.exists(flair_path):
        print(f"❌ Flair 파일 없음: {flair_path}")
        continue
    if not os.path.exists(mask_path):
        print(f"❌ Mask 파일 없음: {mask_path}")
        continue

    flair_img = nib.load(flair_path).get_fdata()
    mask_img = nib.load(mask_path).get_fdata()

    for i in range(flair_img.shape[2]):
        flair_slice = flair_img[:, :, i]
        mask_slice = mask_img[:, :, i]

        if np.max(mask_slice) == 0:
            continue

        try:
            flair_slice = normalize(flair_slice)
            mask_slice = (mask_slice > 0).astype(np.uint8) * 255

            imageio.imwrite(os.path.join(output_image_dir, f"{folder}_slice_{i}.png"), flair_slice)
            imageio.imwrite(os.path.join(output_mask_dir, f"{folder}_slice_{i}.png"), mask_slice)
        except Exception as e:
            print(f"❌ 저장 실패: {folder} slice {i} - {e}")
