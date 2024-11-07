import os
import random
from PIL import Image, ImageDraw
import numpy as np

def create_random_mask(image_size, mask_size):
    """指定されたサイズでランダムな矩形マスクを生成する"""
    mask = Image.new("L", image_size, 0)  # 黒のマスク
    draw = ImageDraw.Draw(mask)
    x = random.randint(0, image_size[0] - mask_size[0])
    y = random.randint(0, image_size[1] - mask_size[1])
    draw.rectangle((x, y, x + mask_size[0], y + mask_size[1]), fill=255)  # 白い矩形を描画
    return mask

def create_missing_images(src_folder, dest_folder, num_samples, mask_size=(16, 16)):
    """指定数の画像をランダムに選択して、欠損部分を作成し保存する"""
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # フォルダ内の画像を取得し、指定数だけランダムに選択
    images = [f for f in os.listdir(src_folder) if f.endswith(".png")]
    
    # フォルダ内の画像数を確認し、`num_samples`が多すぎる場合に調整
    if num_samples > len(images):
        print(f"警告: サンプル数がフォルダ内の画像数より多いため、{len(images)}に調整します。")
        num_samples = len(images)
        
    selected_images = random.sample(images, num_samples)

    for img_name in images:
        # 画像を読み込み
        img_path = os.path.join(src_folder, img_name)
        img = Image.open(img_path).convert("RGBA")
        
        # マスクを生成
        mask = create_random_mask(img.size, mask_size)
        
        # マスクを適用して欠損画像を作成
        img_array = np.array(img)
        mask_array = np.array(mask)
        
        # マスクで指定された部分を透明にする
        img_array[mask_array == 255] = (0, 0, 0, 0)  # 欠損部分を透明にする
        
        # 新しい画像を保存
        missing_img = Image.fromarray(img_array, "RGBA")
        missing_img.save(os.path.join(dest_folder, img_name))
        
        # マスク画像も保存
        mask.save(os.path.join(mask_folder, img_name))

# 使用例
src_folder = "/Users/chinq500/Desktop/archive/Skins/"  # 元の画像フォルダ
dest_folder = "/Users/chinq500/Desktop/archive/Dest/"  # 欠損画像の保存フォルダ
mask_folder = "/Users/chinq500/Desktop/archive/Masks/"  # マスク画像の保存フォルダ
num_samples = 2  # ランダムに選択する画像数
mask_size = (16, 16)  # 欠損部分のサイズ

create_missing_images(src_folder, dest_folder, num_samples, mask_size)
