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

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import matplotlib as mpl

# 日本語フォントの設定
# Windows の場合
plt.rcParams['font.family'] = 'MS Gothic'  # Windows 標準の日本語フォント
# あるいは IPAフォントなどがインストール済みの場合
# plt.rcParams['font.family'] = 'IPAGothic'

# Mac の場合はコメントアウトを外す
# plt.rcParams['font.family'] = 'Hiragino Sans GB'

# Linux の場合はコメントアウトを外す
# plt.rcParams['font.family'] = 'Noto Sans CJK JP'

# ディレクトリパス設定
skins_dir = 'C:/Users/Owner/Desktop/archive/Skins'
missing_dir = 'C:/Users/Owner/Desktop/archive/Missing'
masks_dir = 'C:/Users/Owner/Desktop/archive/Masks'
output_dir = 'C:/Users/Owner/Desktop/archive/Results'

# 出力ディレクトリが存在しない場合は作成
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 画像読み込み関数
def load_image(path, channels=4):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=channels)
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]に正規化
    return image

# 推論用サンプルの準備
def prepare_inference_sample(file_name):
    # ファイル名からパスを生成
    missing_file = tf.strings.join(["missing_", file_name])
    missing_path = tf.strings.join([missing_dir, missing_file], separator='/')
    mask_file = tf.strings.join(["mask_", file_name])
    mask_path = tf.strings.join([masks_dir, mask_file], separator='/')
    
    # 画像とマスクを読み込み
    missing = load_image(missing_path, channels=4)
    mask = load_image(mask_path, channels=1)
    
    # 入力は欠損画像とマスクをチャネル方向に連結
    input_image = tf.concat([missing, mask], axis=-1)
    
    # バッチ次元を追加
    input_image = tf.expand_dims(input_image, 0)
    
    return input_image, file_name

# 保存された学習済みモデルを読み込む
# model = load_model('U_NET/UNET_50000.h5')
# model = load_model('U_NET/UNET.h5', custom_objects={"total_loss": total_loss})
models_dir = {}
models_dir["l1"] = load_model("models/l1_loss.h5", custom_objects={"l1_loss": l1_loss})
models_dir["l2"] = load_model("models/l2_loss.h5", custom_objects={"l2_loss": l2_loss})
models_dir["edge"] = load_model("models/edge_loss.h5", custom_objects={"edge_loss": edge_loss})
models_dir["mae"] = load_model("models/mae_loss.h5", custom_objects={"mae_loss": mae_loss})
models_dir["mae_2"] = load_model("models/mae_2_loss.h5")
models_dir["mse"] = load_model("models/mse_loss.h5", custom_objects={"mse_loss": mse_loss})
models_dir["mse_2"] = load_model("models/mse_2_loss.h5")
models_dir["ssim"] = load_model("models/ssim_loss.h5", custom_objects={"ssim_loss": ssim_loss})
models_dir["total"] = load_model("models/total_loss.h5", custom_objects={"total_loss": total_loss})
# models_dir["all"] = model = load_model('U_NET/UNET.h5', custom_objects={'total_loss': total_loss, "edge_loss": edge_loss, "l1_loss": l1_loss, "l2_loss": l2_loss, "mae_loss": mae_loss, "mse_loss": mse_loss, "ssim_loss": ssim_loss, "laplacian_loss": laplacian_loss})

# テスト用画像ファイル名のリストを取得
test_files = os.listdir(skins_dir)
test_files = [f for f in test_files if f.endswith('.png')]

# テスト画像数の制限（必要に応じて）
test_limit = 50
test_files = test_files[:test_limit]

# 単一画像の推論を行う関数（特定の画像に対して処理したい場合）
def infer_single_image(filename):
    # 推論用サンプルを準備
    input_image, _ = prepare_inference_sample(filename)
    
    # # 推論実行
    # predicted_skin = model.predict(input_image)[0]

    predicted_skins = []
    for model in models_dir.values():
        predicted_skins.append(model.predict(input_image)[0])
    
    # 元の画像も読み込む（比較用）
    missing_path = os.path.join(missing_dir, f"missing_{filename}")
    missing_image = load_image(missing_path).numpy()
    
    original_path = os.path.join(skins_dir, filename)
    original_image = load_image(original_path).numpy()
    
    # 結果の可視化と表示
    # plt.figure(figsize=(15, 5))
    fig, axes = plt.subplots(1, 11, figsize=(18, 2))
    
    # 元の画像
    axes[0].imshow(original_image)
    axes[0].set_title('元画像', fontsize=14)
    axes[0].axis('off')

    # 欠損画像
    axes[1].imshow(missing_image)
    axes[1].set_title('欠損画像', fontsize=14)
    axes[1].axis('off')

    for i, (key, predicted_skin) in enumerate(zip(models_dir.keys(), predicted_skins)):
        axes[i+2].imshow(predicted_skin)
        axes[i+2].set_title(f'{key}損失', fontsize=14)
        axes[i+2].axis('off')
    
    # # タイトルがきちんと表示されるよう余白調整
    # plt.tight_layout()

    plt.show()
    
    return predicted_skins

# メイン実行部分
if __name__ == "__main__":
    # フォント設定の確認
    print("利用可能なフォント:")
    print(mpl.font_manager.findSystemFonts(fontpaths=None, fontext='ttf'))

    # 全てのテスト画像で推論を実行
    print("個別推論処理を開始します...")
    for file_name in test_files:
        print(f"処理中: {file_name}")
        infer_single_image(file_name)
    
    # または、より効率的なバッチ処理で推論
    # print("バッチ推論処理を開始します...")
    # batch_inference(batch_size=16)
    
    print("推論処理が完了しました。結果は以下のディレクトリに保存されています:")
    print(output_dir)
