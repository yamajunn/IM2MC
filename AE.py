import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# 画像データセットを定義するクラス
class CustomImageDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGBA')  # RGBA画像を読み込み
        if self.transform:
            image = self.transform(image)
        return image

# オートエンコーダモデルの定義
class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),  # RGBAのため4チャンネル
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # プーリングでサイズを半分に
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2),  # サイズを元に戻す
            nn.ReLU(),
            nn.ConvTranspose2d(16, 4, kernel_size=2, stride=2),
            nn.Sigmoid()  # 出力を0-1に制限
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 学習関数
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        for i, inputs in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)  # 出力と入力の形状が同じ
            loss.backward()
            optimizer.step()
            print(f"{i}/{len(dataloader)}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# メイン処理
if __name__ == "__main__":
    # データセットの準備
    image_dir = '/Users/chinq500/Desktop/archive/Skins'  # 画像フォルダのパスを指定
    all_image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
    
    # ランダムに10000枚を選択
    random_sample = random.sample(all_image_files, 10000)

    # データローダーの準備
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    dataset = CustomImageDataset(image_files=random_sample, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # モデル、損失関数、オプティマイザの初期化
    model = SimpleAutoencoder()
    criterion = nn.MSELoss()  # 平均二乗誤差損失
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # モデルの学習
    train_model(model, dataloader, criterion, optimizer, num_epochs=10)

    # 学習したモデルの保存
    torch.save(model.state_dict(), 'trained_autoencoder.pth')
    print("モデルが 'trained_autoencoder.pth' として保存されました。")
