import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os

# U-Net モデルの定義
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # エンコーダ部分
        self.encoder1 = self.conv_block(3, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        # ボトム部分
        self.center = self.conv_block(512, 1024)
        
        # デコーダ部分
        self.decoder4 = self.conv_block(1024, 512)
        self.decoder3 = self.conv_block(512, 256)
        self.decoder2 = self.conv_block(256, 128)
        self.decoder1 = self.conv_block(128, 64)
        
        # 出力層
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        # エンコーダ
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        
        # ボトム部分
        center = self.center(enc4)
        
        # デコーダ
        dec4 = self.decoder4(center)
        dec3 = self.decoder3(dec4)
        dec2 = self.decoder2(dec3)
        dec1 = self.decoder1(dec2)
        
        # 出力
        output = self.final_conv(dec1)
        return output

# モデルインスタンス
model = UNet().cuda()

# データセットクラス（補完画像を訓練するためのもの）
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, complete_dir, missing_dir, mask_dir, transform=None):
        self.complete_images = [os.path.join(complete_dir, f) for f in os.listdir(complete_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
        self.missing_images = [os.path.join(missing_dir, f) for f in os.listdir(missing_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
        self.mask_images = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
        
        self.transform = transform

    def __len__(self):
        return len(self.complete_images)

    def __getitem__(self, idx):
        complete_img = Image.open(self.complete_images[idx]).convert("RGB")
        missing_img = Image.open(self.missing_images[idx]).convert("RGB")
        mask_img = Image.open(self.mask_images[idx]).convert("L")  # モノクロでマスクを表現
        
        if self.transform:
            complete_img = self.transform(complete_img)
            missing_img = self.transform(missing_img)
            mask_img = self.transform(mask_img)
        
        return missing_img, mask_img, complete_img

# データの前処理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# データセットとデータローダーの作成
complete_dir = "path/to/complete/images"
missing_dir = "path/to/missing/images"
mask_dir = "path/to/mask/images"

dataset = ImageDataset(complete_dir, missing_dir, mask_dir, transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# ロス関数と最適化手法
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 訓練ループ
num_epochs = 20
for epoch in range(num_epochs):
    for missing_img, mask_img, complete_img in dataloader:
        missing_img = missing_img.cuda()
        complete_img = complete_img.cuda()

        # モデルの出力を計算
        output = model(missing_img)

        # ロスの計算と最適化
        loss = criterion(output, complete_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
