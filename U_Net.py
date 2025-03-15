import tensorflow as tf
import cv2
import numpy as np

# 学習済みモデルの読み込み
model = tf.keras.models.load_model('U_NET/UNET.h5')

# 画像とマスクの読み込み
image = cv2.imread('Skins/0_missing.png', cv2.IMREAD_UNCHANGED)  # RGBA画像
mask = cv2.imread('Skins/0_mask.png', cv2.IMREAD_GRAYSCALE)  # グレースケール

# 画像の前処理
mask = np.expand_dims(mask, axis=-1)  # チャンネル次元を追加

# 入力データ作成
masked_image = image * (1 - mask)  # 欠損部分をゼロにする
input_data = np.concatenate([image, mask], axis=-1)  # (64,64,5)
input_data = np.expand_dims(input_data, axis=0)  # バッチ次元追加

# モデルによる補完
output = model.predict(input_data)
output = np.squeeze(output, axis=0)  # バッチ次元を削除

# 結果の保存（RGBA変換）
output = (output * 255).astype(np.uint8)
cv2.imwrite('Output/output.png', output)

print("補完画像を保存しました: Output/output.png")
