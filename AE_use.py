import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn

# オートエンコーダモデルの定義（学習時と同じモデル構造を使用）
class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 4, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 保存したモデルを読み込む関数
def load_model(model_path):
    model = SimpleAutoencoder()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 推論モードに切り替え
    return model

# 推論処理
def predict_image(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    # 画像の読み込みと前処理
    image = Image.open(image_path).convert('RGBA')
    input_tensor = transform(image).unsqueeze(0)  # バッチ次元を追加

    # モデルによる予測
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # 出力を画像として変換
    output_image = transforms.ToPILImage()(output_tensor.squeeze(0))
    return image, output_image  # 入力画像と出力画像を返す

# メイン処理
if __name__ == "__main__":
    # モデルの読み込み
    model_path = 'trained_autoencoder.pth'
    model = load_model(model_path)

    # 使用する画像のパス
    test_image_path = '1_.png'

    # 推論の実行
    original_image, reconstructed_image = predict_image(model, test_image_path)

    # 結果の表示
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(reconstructed_image)
    ax[1].set_title("Reconstructed Image")
    ax[1].axis("off")

    plt.show()
