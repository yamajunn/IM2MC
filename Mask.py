import os
import numpy as np
from PIL import Image
import random

# Mask作成関数 (矩形範囲を隠す)
def create_mask(image_size, mask_type='rectangular', mask_ratio=0.4):
    """
    隠す範囲を矩形で作成する関数。
    mask_ratio: 隠す領域の割合（例: 0.4は40%の領域）
    """
    mask = np.ones(image_size, dtype=np.uint8)  # 最初は全て1（可視部分）
    
    if mask_type == 'rectangular':
        # 隠す領域のサイズを計算
        area = int(image_size[0] * image_size[1] * mask_ratio)
        height = random.randint(10, int(image_size[0] / 2))  # 隠す範囲の高さ（10〜画像サイズの半分）
        width = int(area / height)  # 面積を基に幅を計算
        
        # width と height が画像のサイズを超えないように調整
        width = min(width, image_size[1])
        height = min(height, image_size[0])
        
        # 隠す領域の位置をランダムに決定
        x_start = random.randint(0, image_size[1] - width)  # x軸の開始位置
        y_start = random.randint(0, image_size[0] - height)  # y軸の開始位置
        
        # マスクを作成（指定した範囲を0にする）
        mask[y_start:y_start + height, x_start:x_start + width] = 0
    
    return mask

# Missing画像作成関数
def create_missing_image(original_image, mask):
    """
    元の画像からマスクを適用してMissing画像を作成。
    mask: 0が隠すべき部分、1が可視部分
    """
    missing_image = np.array(original_image)
    missing_image[mask == 0] = 0  # マスクされた部分を0（透明）にする
    
    return Image.fromarray(missing_image)

# 画像を一括で処理する関数
def process_images(input_folder, output_mask_folder, output_missing_folder, mask_type='rectangular', mask_ratio=0.4):
    # 入力フォルダ内の画像ファイルをリスト化
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))]
    
    # 出力フォルダが存在しない場合は作成
    os.makedirs(output_mask_folder, exist_ok=True)
    os.makedirs(output_missing_folder, exist_ok=True)
    
    # 各画像に対して処理を行う
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        
        # 画像を開く
        original_image = Image.open(image_path).convert("RGBA")
        image_size = original_image.size  # (64, 64)
        
        # マスクを作成
        mask = create_mask(image_size, mask_type=mask_type, mask_ratio=mask_ratio)
        
        # Missing画像を作成
        missing_image = create_missing_image(original_image, mask)
        
        # マスク画像を作成（可視領域が1、隠す部分が0）
        mask_image = Image.fromarray(mask * 255).convert("L")  # 0/255にする
        
        # 出力ファイル名を設定
        mask_filename = f"mask_{image_file}"
        missing_filename = f"missing_{image_file}"
        
        # 画像を保存
        mask_image.save(os.path.join(output_mask_folder, mask_filename))
        missing_image.save(os.path.join(output_missing_folder, missing_filename))

# 使用例
input_folder = "C:/Users/Owner/Desktop/archive/Skins/"  # 入力フォルダのパスを指定
output_mask_folder = "C:/Users/Owner/Desktop/archive/Masks/"  # マスク画像保存用フォルダのパス
output_missing_folder = "C:/Users/Owner/Desktop/archive/Missing/"  # Missing画像保存用フォルダのパス

# 画像処理を実行
process_images(input_folder, output_mask_folder, output_missing_folder, mask_type='rectangular', mask_ratio=0.4)
