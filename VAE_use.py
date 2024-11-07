import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from sklearn.cluster import KMeans

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
            nn.Sigmoid(),  # 出力が0〜1の範囲に収まる
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
# 画像補完関数の定義
# -------------------------
def inpaint_image(model, missing_img, mask_img, device):
    model.eval()  # 推論モードに切り替え
    
    # 入力データを作成してVAEに送る
    missing_img = missing_img.to(device).unsqueeze(0)  # バッチ次元を追加
    mask_img = mask_img.to(device).unsqueeze(0)
    
    # マスクを含む欠損画像をVAEに入力
    input_img = torch.cat((missing_img, mask_img), dim=1)  # 5チャネルにする
    with torch.no_grad():
        reconstructed_img, _, _ = model(input_img)
    
    # 出力画像のバッチ次元を除去して返す
    reconstructed_img = reconstructed_img.squeeze(0).cpu()
    return reconstructed_img

# -------------------------
# 色数を制限するポストプロセス関数の定義
# -------------------------
def reduce_colors(img_tensor, num_colors=16):
    img_np = img_tensor.permute(1, 2, 0).numpy() * 255
    img_np = img_np.reshape(-1, 4)  # 各ピクセルを1次元化 (num_pixels, 4)

    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    labels = kmeans.fit_predict(img_np[:, :3])  # RGB部分のみに適用

    new_colors = kmeans.cluster_centers_[labels].astype("uint8")
    img_np[:, :3] = new_colors  # RGBチャネルのみ更新

    img_np = img_np.reshape(64, 64, 4).astype("uint8")
    img_pil = Image.fromarray(img_np, "RGBA")
    return img_pil

# -------------------------
# 画像補完の実行とマスク部分の合成
# -------------------------
def combine_with_mask(original_img, reconstructed_img, mask_img):
    mask_np = mask_img.squeeze().numpy()  # マスク画像をNumPyに変換
    reconstructed_np = reconstructed_img.permute(1, 2, 0).numpy()  # 補完画像をNumPyに変換
    original_np = original_img.permute(1, 2, 0).numpy()  # 元の欠損画像をNumPyに変換

    # マスク部分のみ補完画像を適用
    combined_img_np = np.where(mask_np[..., None] > 0.5, reconstructed_np, original_np)
    
    combined_img_pil = Image.fromarray((combined_img_np * 255).astype(np.uint8), "RGBA")
    return combined_img_pil

# -------------------------
# テスト用の画像の読み込みと変換
# -------------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

missing_img_path = "0.png"  # 欠損画像のパス
mask_img_path = "0_mask.png"        # マスク画像のパス

missing_img = Image.open(missing_img_path).convert("RGBA")
mask_img = Image.open(mask_img_path).convert("L")  # マスク画像は1チャネル

missing_img = transform(missing_img)
mask_img = transform(mask_img)

# -------------------------
# モデルの準備と画像補完の実行
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 128

model = VAE(latent_dim=latent_dim).to(device)
model_path = "model_checkpoints/vae_model_epoch_100.pth"
model.load_state_dict(torch.load(model_path))
model.eval()

reconstructed_img = inpaint_image(model, missing_img, mask_img, device)
reconstructed_img_pil = reduce_colors(reconstructed_img, num_colors=64)

# 元画像と補完結果をマスクに従って合成
completed_img_pil = combine_with_mask(missing_img, reconstructed_img, mask_img)

# -------------------------
# 補完された画像を保存
# -------------------------
completed_img_pil.save("completed_image_with_mask.png")
