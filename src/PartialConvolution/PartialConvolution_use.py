import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

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
    

# 軽量オートエンコーダ（Partial Convolution使用）の定義
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
        x = self.relu1(x)
        
        # Partial Conv + ReLU
        x, mask = self.partial_conv2(x, mask)
        x = self.relu2(x)
        
        # デコーダ処理
        x = self.decoder(x)
        return x

# 画像補完関数
def complete_image(model, input_image_path, mask_image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 画像とマスクの読み込み
    input_image = Image.open(input_image_path).convert('RGBA')
    mask_image = Image.open(mask_image_path).convert('L')  # グレースケールに変換
    
    # マスクをRGBA形式に変換
    mask_image_rgba = Image.merge("RGBA", (mask_image, mask_image, mask_image, mask_image))

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    input_tensor = transform(input_image).unsqueeze(0).to(device)  # バッチ次元を追加
    mask_tensor = transform(mask_image_rgba).unsqueeze(0).to(device)  # RGBAに変換したマスクをバッチ次元を追加

    # モデルを評価モードに設定
    model.eval()
    
    with torch.no_grad():
        output = model(input_tensor, mask_tensor)  # モデルの予測

    # 元の画像と生成された画像を組み合わせる
    input_image_tensor = input_tensor.squeeze(0)  # (C, H, W)に戻す
    output_image_tensor = output.squeeze(0)  # (C, H, W)に戻す

    # マスクを2次元にし、欠損部分を補完された画像で上書き
    mask_2d = mask_tensor.squeeze(0).cpu()  # (1, 64, 64)に変換
    mask_2d = (mask_2d == 0).float()  # 0の部分を1に、それ以外を0にする

    # 補完された画像を作成
    completed_image_tensor = input_image_tensor * (1 - mask_2d) + output_image_tensor * mask_2d

    # 補完された画像をPILに変換して保存
    completed_image_pil = transforms.ToPILImage()(completed_image_tensor.cpu())
    completed_image_pil.save("completed_image.png")
    
    return input_image, completed_image_pil


# メイン処理
if __name__ == "__main__":
    input_image_path = "1_.png"  # 入力画像のパスを指定
    mask_image_path = "1__.png"  # マスク画像のパスを指定
    model_path = "lightweight_partial_autoencoder.pth"  # モデルのパスを指定

    # モデルの初期化と重みの読み込み
    model = LightweightPartialAutoencoder()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # 画像補完処理の実行
    original_image, completed_image = complete_image(model, input_image_path, mask_image_path)