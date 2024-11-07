import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
import random

# 軽量Partial Convolutional Layer
class PartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(PartialConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        
        # mask_convの重みを固定してバイアスを無効化
        nn.init.constant_(self.mask_conv.weight, 1.0)
        self.mask_conv.weight.requires_grad = False

    def forward(self, x, mask):
        output = self.conv(x * mask)
        mask_output = self.mask_conv(mask)
        
        mask_output = torch.clamp(mask_output, min=1e-8)
        output = output / mask_output
        new_mask = (mask_output > 0).float()
        
        return output, new_mask

# 軽量オートエンコーダ（Partial Convolution使用）
class LightweightPartialAutoencoder(nn.Module):
    def __init__(self):
        super(LightweightPartialAutoencoder, self).__init__()
        self.partial_conv1 = PartialConv2d(4, 32, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.partial_conv2 = PartialConv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # 出力を0-1に制限
        )

    def forward(self, x, mask):
        # Partial Conv + ReLU
        x, mask = self.partial_conv1(x, mask)
        x = self.relu1(x)  # ReLUにはマスクは不要
        
        # Partial Conv + ReLU
        x, mask = self.partial_conv2(x, mask)
        x = self.relu2(x)  # ReLUにはマスクは不要
        
        # デコーダ処理
        x = self.decoder(x)
        return x

# データセット読み込み（RGBA対応）
class RGBAImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGBA')  # RGBA形式で読み込む
        
        if self.transform:
            image = self.transform(image)

        return image

# マスク生成関数
def create_mask(image_shape):
    mask = np.ones(image_shape, dtype=np.float32)
    h, w = image_shape[1], image_shape[2]
    mask_size = random.randint(h // 4, h // 2)
    x1 = random.randint(0, h - mask_size)
    y1 = random.randint(0, w - mask_size)
    x2 = x1 + mask_size
    y2 = y1 + mask_size
    mask[:, x1:x2, y1:y2] = 0
    return mask

# トレーニングの設定
def train_model(model, dataloader, num_epochs=10, learning_rate=1e-4, save_path='lightweight_partial_autoencoder.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, images in enumerate(dataloader):
            images = images.to(device)
            
            # 欠損マスクの作成
            masks = torch.tensor(create_mask(images.shape)).float().to(device)
            
            # 欠損部分を適用
            masked_images = images * masks

            # モデル出力と損失計算
            outputs = model(masked_images, masks)
            loss = loss_fn(outputs, images)  # 元画像との誤差を計算
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(dataloader):
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}] 完了, 平均Loss: {avg_epoch_loss:.4f}')

    # モデルの保存
    torch.save(model.state_dict(), save_path)
    print(f"モデルが '{save_path}' に保存されました。")

# メイン処理
if __name__ == "__main__":
    batch_size = 16
    num_epochs = 3
    learning_rate = 1e-4
    save_path = '_.pth'

    # データ前処理の定義
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # データセットの作成（RGBA画像対応）
    dataset = RGBAImageDataset(root_dir='/Users/chinq500/Desktop/archive/Skins', transform=transform)
    
    # データローダーの作成
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # モデルの初期化
    model = LightweightPartialAutoencoder()

    # モデルのトレーニング
    train_model(model, dataloader, num_epochs, learning_rate, save_path)
