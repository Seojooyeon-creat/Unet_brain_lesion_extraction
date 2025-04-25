import os
import random
import shutil

# 원본 경로
img_dir = 'BraTS_slices/images'
mask_dir = 'BraTS_slices/masks'

# 새 경로
new_img_dir = 'BraTS_slices_subset/images'
new_mask_dir = 'BraTS_slices_subset/masks'

os.makedirs(new_img_dir, exist_ok=True)
os.makedirs(new_mask_dir, exist_ok=True)

# 이미지 파일 목록 (마스크랑 이름 동일해야 함)
images = sorted([f for f in os.listdir(img_dir) if f in os.listdir(mask_dir)])

#n개 무작위 추출 (n값 조절해야됌)
selected = random.sample(images, n)

# 복사
for fname in selected:
    shutil.copy(os.path.join(img_dir, fname), os.path.join(new_img_dir, fname))
    shutil.copy(os.path.join(mask_dir, fname), os.path.join(new_mask_dir, fname))

print("완료")
