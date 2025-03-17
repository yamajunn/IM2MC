from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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

# 画像を読み込む
image_path = 'Skins/0.png'
image = Image.open(image_path).convert('RGBA')

# 画像をTensorFlowのテンソルに変換
image = np.array(image)
image = tf.constant(image, dtype=tf.float32)
edges = laplacian_filter(image)

# 画像を表示する
plt.imshow(edges, cmap='gray')  # グレースケールで表示
plt.axis('off')  # 軸を非表示にする
plt.show()

# h, w = image.size
# print(h, w)
# # 画像をnumpy配列に変換
# image_array = np.array(image)

# # 新しい画像を格納する配列を作成
# new_image_array = np.zeros((h, w))

# # 各ピクセルの絶対値差分の平均を計算
# for i in range(1, h-1):
#     for j in range(1, w-1):
#         diff_sum = 0
#         count = 0
#         for x in range(-1, 2):
#             for y in range(-1, 2):
#                 if x != 0 or y != 0:
#                     diff_sum += abs(int(image_array[i, j]) - int(image_array[i + x, j + y]))
#                     count += 1
#         new_image_array[i, j] = diff_sum / count

# # 新しい画像をPIL Imageに変換
# new_image = Image.fromarray(new_image_array.astype(np.uint8))

# # 画像を表示する
# plt.imshow(new_image, cmap='gray')  # グレースケールで表示
# plt.axis('off')  # 軸を非表示にする
# plt.show()

# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt

# # 画像を読み込む
# image_path = 'Skins/0_mask.png'
# image = Image.open(image_path).convert('RGBA')  # RGBA形式で読み込む

# h, w = image.size
# print(h, w)
# # 画像をnumpy配列に変換
# image_array = np.array(image)

# # 新しい画像を格納する配列を作成
# new_image_array = np.zeros((h, w, 4))

# # 各ピクセルの絶対値差分の平均を計算
# for i in range(1, h-1):
#     for j in range(1, w-1):
#         for c in range(4):  # 各チャンネルに対して処理を行う
#             diff_sum = 0
#             count = 0
#             for x in range(-1, 2):
#                 for y in range(-1, 2):
#                     if x != 0 or y != 0:
#                         diff_sum += abs(int(image_array[i, j, c]) - int(image_array[i + x, j + y, c]))
#                         count += 1
#             new_image_array[i, j, c] = diff_sum / count

# # 新しい画像をPIL Imageに変換
# new_image = Image.fromarray(new_image_array.astype(np.uint8))

# # 画像を表示する
# plt.imshow(new_image)
# plt.axis('off')  # 軸を非表示にする
# plt.show()
