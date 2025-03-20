import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Loss functions
def generator_loss(disc_generated_output, gen_output, target, lambda_l1=100):
    # Adversarial loss
    gan_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(disc_generated_output), disc_generated_output)
    
    # L1 loss (pixel-wise difference)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    
    # Total loss
    total_loss = gan_loss + (lambda_l1 * l1_loss)
    
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
model = load_model("models/GAN/checkpoint/pixelart_inpainting_generator_epoch_210.h5", custom_objects={'generator_loss': generator_loss, "discriminator_loss": discriminator_loss})
# model = load_model("models/laplacian_loss.h5", custom_objects={"laplacian_loss": laplacian_loss})

predicted_image = model.predict(input_image)[0]

# 予測結果を表示

plt.figure(figsize=(6, 6)).set(facecolor='gray')
plt.imshow(predicted_image)
plt.title('Predicted Image')
plt.axis('off')
plt.show()