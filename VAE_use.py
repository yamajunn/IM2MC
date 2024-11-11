import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# U-Net構造にスキップ接続とAttentionを持つVAEモデル
# -------------------------
class AttentionBlock(nn.Module):
    """Self-attention block to capture spatial dependencies."""
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        value = self.value_conv(x).view(batch_size, -1, width * height)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

class UNetVAE(nn.Module):
    def __init__(self, latent_dim=128, num_attention_blocks=3):
        super(UNetVAE, self).__init__()
        
        # Encoder部分
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 32, 4, stride=2, padding=1),  # encoder.0
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # encoder.2
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # encoder.4
            nn.ReLU()
        )

        # Attention Blocks
        self.attention_blocks = nn.Sequential(
            *[AttentionBlock(128) for _ in range(num_attention_blocks)]
        )

        # 潜在変数用の線形層
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)
        
        # デコードのための線形層
        self.fc_decode = nn.Linear(latent_dim, 128 * 8 * 8)
        
        # Decoder部分
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # decoder.0
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # decoder.2
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, 4, stride=2, padding=1),  # decoder.4
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = self.attention_blocks(x)  # 複数のAttention Blockを追加
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) * 1.5  # ランダム性を強化
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
# ガイド付き損失関数
# -------------------------
def guided_inpainting_loss(recon_x, x, mask, mu, logvar):
    bce_loss = nn.functional.mse_loss(recon_x * mask, x * mask, reduction='sum')  # マスク内の再構成誤差
    context_guide_loss = nn.functional.mse_loss(recon_x * (1 - mask), x * (1 - mask), reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce_loss + context_guide_loss + kld_loss

# -------------------------
# 画像補完関数
# -------------------------
def inpaint_image(model, missing_img, mask_img, device, context_size=10):
    model.eval()
    missing_img = missing_img.to(device).unsqueeze(0)
    mask_img = mask_img.to(device).unsqueeze(0)
    input_img = torch.cat((missing_img, mask_img), dim=1)
    
    with torch.no_grad():
        reconstructed_img, _, _ = model(input_img)
    
    # マスク部分を周囲のピクセルに基づいて補完
    inpainted_img = missing_img.clone()
    padding = context_size // 2
    mask_dilated = F.conv2d(F.pad(mask_img, (padding, padding, padding, padding)), torch.ones(1, 1, context_size, context_size, device=device), padding=0)
    mask_dilated = (mask_dilated > 0).float()
    mask_dilated = F.interpolate(mask_dilated, size=missing_img.shape[2:], mode='nearest')  # サイズを一致させる
    
    # 周囲のピクセルに基づいて補完
    context_mask = mask_dilated - mask_img
    inpainted_img = inpainted_img * (1 - mask_img) + reconstructed_img * mask_img
    inpainted_img = inpainted_img * (1 - context_mask) + missing_img * context_mask
    
    return inpainted_img.squeeze(0).cpu()

# -------------------------
# 実行の準備
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 128
model = UNetVAE(latent_dim=latent_dim).to(device)

# 学習済みモデルのロード
checkpoint = torch.load("model_checkpoints/vae_model_epoch_100.pth", weights_only=True)
model.load_state_dict(checkpoint, strict=False)  # strict=Falseで新しいパラメータを初期化
model.eval()

# マスク画像、欠損画像を読み込み、補完実行
missing_img = Image.open("1.png").convert("RGBA")
mask_img = Image.open("1_mask.png").convert("L")
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
missing_img = transform(missing_img)
mask_img = transform(mask_img)

# 補完
reconstructed_img = inpaint_image(model, missing_img, mask_img, device, context_size=10)
reconstructed_img_pil = transforms.ToPILImage()(reconstructed_img)
reconstructed_img_pil.save("completed_image.png")
