import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import os
import time

from model.dataset import BrainTumorDataset
from model.unet import UNet
from visualize import visualize

# ✅ CPU 스레드 최대로 사용 (M1/M2/M4 최적화)
torch.set_num_threads(os.cpu_count())
print(f"🔧 CPU 최대 스레드 설정: {torch.get_num_threads()}")

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 전처리
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

# 데이터셋
dataset = BrainTumorDataset("BraTS_slices_subset/images", "BraTS_slices_subset/masks", transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)  # ← macOS는 0 유지 권장

# 모델 및 학습 설정
model = UNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 학습 루프
for epoch in range(10):
    start_time = time.time()
    model.train()
    total_loss = 0
    for batch_idx, (imgs, masks) in enumerate(dataloader):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Batch {batch_idx+1}/{len(dataloader)} Loss: {loss.item():.4f}")

    print(f"✅ Epoch [{epoch+1}/10] - Total Loss: {total_loss:.4f}")

end_time = time.time()
duration = end_time - start_time

# 예측 시각화
model.eval()
with torch.no_grad():
    img, mask = dataset[0]
    pred = model(img.unsqueeze(0).to(device))
    visualize(img, mask, pred)


torch.save(model.state_dict(), "unet_brats.pth")
