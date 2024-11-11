import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm  # tqdmをインポート

# -------------------------
# データセットの定義（サンプル数を絞る）
# -------------------------
class ImageDataset(Dataset):
    def __init__(self, complete_dir, missing_dir, mask_dir, transform=None, max_samples=None):
        valid_extensions = (".png", ".jpg", ".jpeg", ".bmp")

        # ディレクトリから画像ファイルのみを収集
        self.complete_images = [os.path.join(complete_dir, f) for f in os.listdir(complete_dir) if f.lower().endswith(valid_extensions)]
        self.missing_images = [os.path.join(missing_dir, f) for f in os.listdir(missing_dir) if f.lower().endswith(valid_extensions)]
        self.mask_images = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.lower().endswith(valid_extensions)]

        if max_samples is not None:
            self.complete_images = self.complete_images[:max_samples]
            self.missing_images = self.missing_images[:max_samples]
            self.mask_images = self.mask_images[:max_samples]

        self.transform = transform

    def __len__(self):
        return len(self.complete_images)

    def __getitem__(self, idx):
        complete_img = Image.open(self.complete_images[idx]).convert("RGBA")
        missing_img = Image.open(self.missing_images[idx]).convert("RGBA")
        mask_img = Image.open(self.mask_images[idx]).convert("L")  # マスクは1チャネル

        if self.transform:
            complete_img = self.transform(complete_img)
            missing_img = self.transform(missing_img)
            mask_img = self.transform(mask_img)

        missing_img = torch.cat((missing_img, mask_img), dim=0)
        
        return missing_img, complete_img

# -------------------------
# VAEモデルの定義
# -------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.fc_decode(z)
        z = z.view(z.size(0), 128, 8, 8)
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# -------------------------
# 学習ループの定義
# -------------------------
def vae_loss(recon_x, x, mu, logvar):
    bce_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce_loss + kld_loss

def train(model, dataloader, optimizer, num_epochs=10, save_interval=2):
    model.train()

    # 保存ディレクトリの作成
    os.makedirs("model_checkpoints", exist_ok=True)

    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100, leave=False)
        for missing_img, complete_img in progress_bar:
            missing_img = missing_img.to(device)
            complete_img = complete_img.to(device)
            
            optimizer.zero_grad()
            recon_img, mu, logvar = model(missing_img)
            loss = vae_loss(recon_img, complete_img, mu, logvar)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / len(dataloader.dataset))

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader.dataset)}")
        
        if (epoch + 1) % save_interval == 0:
            model_save_path = f"model_checkpoints/vae_model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"モデルが保存されました: {model_save_path}")

    model_save_path = "vae_model_final.pth"
    torch.save(model.state_dict(), model_save_path)
    print("最終モデルが保存されました:", model_save_path)

# -------------------------
# パラメータの設定と実行
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 256
batch_size = 64
num_epochs = 100
learning_rate = 1e-3

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

max_samples = 940000
dataset = ImageDataset(
                    # complete_dir="/Users/chinq500/Desktop/archive/Skins",
                    # missing_dir="/Users/chinq500/Desktop/archive/Dest",
                    # mask_dir="/Users/chinq500/Desktop/archive/Masks",
                    complete_dir="C:/Users/Owner/Desktop/archive/Skins/",
                    missing_dir="C:/Users/Owner/Desktop/archive/Dest/",
                    mask_dir="C:/Users/Owner/Desktop/archive/Masks/",
                    transform=transform,
                    max_samples=max_samples
                )

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = VAE(latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train(model, dataloader, optimizer, num_epochs=num_epochs)
