import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

import tensorflow as tf
import numpy as np

def l1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def l2_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def edge_loss(y_true, y_pred):
    sobel_true = tf.image.sobel_edges(y_true)
    sobel_pred = tf.image.sobel_edges(y_pred)

    # X方向のエッジ差分
    edge_x_loss = tf.abs(sobel_true[..., 0] - sobel_pred[..., 0])
    # Y方向のエッジ差分
    edge_y_loss = tf.abs(sobel_true[..., 1] - sobel_pred[..., 1])

    # 両方のエッジの差分の平均を損失として使う
    return tf.reduce_mean(edge_x_loss + edge_y_loss)

def mae_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def ssim_loss(y_true, y_pred):
    # SSIMは[0, 1]範囲の画像に適用されるため、出力を[0, 1]に正規化
    y_true = (y_true + 1.0) / 2.0  # RGBA画像などの場合、[-1, 1]の範囲から[0, 1]に変換
    y_pred = (y_pred + 1.0) / 2.0  # 同様に出力を[0, 1]に正規化
    
    # SSIMを計算
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def laplacian_filter(image):
    """RGBAの各チャンネルにラプラシアンフィルタを適用"""
    laplacian_kernel = tf.constant([
        [0,  1,  0],
        [1, -4,  1],
        [0,  1,  0]
    ], dtype=tf.float32)
    
    laplacian_kernel = tf.reshape(laplacian_kernel, [3, 3, 1, 1])  # (高さ, 幅, 入力チャンネル, 出力チャンネル)
    
    # RGBAの各チャンネルに適用するためのフィルタを作成
    filters = tf.tile(laplacian_kernel, [1, 1, 4, 1])  # (3, 3, 4, 4) に拡張

    # 4次元テンソル (バッチ, 高さ, 幅, チャンネル) の形状を維持
    image = tf.expand_dims(image, axis=0)  # バッチ次元を追加 (None, H, W, 4)
    edges = tf.nn.conv2d(image, filters, strides=[1, 1, 1, 1], padding="SAME")

    return tf.squeeze(edges)  # バッチ次元を削除

def laplacian_loss(y_true, y_pred):
    edge_true = laplacian_filter(y_true)
    edge_pred = laplacian_filter(y_pred)

    # L1 損失
    loss = tf.reduce_mean(tf.abs(edge_true - edge_pred))
    return loss

def total_loss(y_true, y_pred):
    loss_l1 = l1_loss(y_true, y_pred)
    # loss_edge = edge_loss(y_true, y_pred)
    # loss_mae = mae_loss(y_true, y_pred)
    # loss_mse = mse_loss(y_true, y_pred)
    # loss_ssim = ssim_loss(y_true, y_pred)
    loss_laplacian = laplacian_loss(y_true, y_pred)

    return loss_l1 + 0.2 * loss_laplacian

# Loss functions
def generator_loss(disc_generated_output, gen_output, target, lambda_l1=100):
    # Adversarial loss
    gan_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(disc_generated_output), disc_generated_output)
    
    # L1 loss (pixel-wise difference)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    loss_ssim = ssim_loss(target, gen_output)
    
    # Total loss
    total_loss = gan_loss + (lambda_l1 * l1_loss) + loss_ssim
    
    return total_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    # Real image loss
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(disc_real_output), disc_real_output)
    
    # Generated image loss
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.zeros_like(disc_generated_output), disc_generated_output)
    
    # Total loss
    total_loss = real_loss + generated_loss
    
    return total_loss

missing_image_path = 'Skins/0_missing.png'
mask_image_path = 'Skins/0_mask.png'

def load_image(path, channels=4):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=channels)
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]に正規化
    return image

missing = load_image(missing_image_path, channels=4)
mask = load_image(mask_image_path, channels=1)

missing = missing * mask

# 入力は欠損画像とマスクをチャネル方向に連結
input_image = tf.concat([missing, mask], axis=-1)

# バッチ次元を追加
input_image = tf.expand_dims(input_image, 0)

# 保存された学習済みモデルを読み込む
model = load_model("models/GAN/GAN_500.h5", custom_objects={'generator_loss': generator_loss, "discriminator_loss": discriminator_loss})
# model = load_model("models/laplacian_loss.h5", custom_objects={"laplacian_loss": laplacian_loss})

predicted_image = model.predict(input_image)[0]

# 予測結果を表示

plt.figure(figsize=(6, 6)).set(facecolor='gray')
plt.imshow(predicted_image)
plt.title('Predicted Image')
plt.axis('off')
plt.show()