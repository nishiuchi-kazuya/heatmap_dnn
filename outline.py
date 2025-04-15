import time
import torch
import numpy as np
import cv2
import argparse
import networkx as nx
from scipy.spatial import distance
import matplotlib.pyplot as plt
from scipy import stats
import os

if __name__ == '__main__':
    device = 'cuda'
    '''
    # ディレクトリ設定
    joint_dataset_dir = 'joint_dataset/'
    parent_dir = '/home/irsl/heatmap_dnn/'  # dataset_jointの親ディレクトリを取得
    output_dir = os.path.join(parent_dir, 'outline_dataset')  # outline_datasetディレクトリのパス
    print(parent_dir)

    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 画像のループ処理（1から30000までの画像を処理）
    #for i in range(0, 31348+1):
    # 1. 画像パスを動的に生成
    #image_path = os.path.join(parent_dir+joint_dataset_dir, f'image_{str(i).zfill(8)}.png')
    print(f"Processing {image_path}")

    # 2. 画像を読み込み
    
    if image_cv is None:
        print(f"Image at path {image_path} は読み込めませんでした。")
        continue
        '''
    image_path='testdata/humanoid9.png'
    image_cv = cv2.imread(image_path)
    # 画像がモノクロかカラーかで処理を分ける
    if len(image_cv.shape) == 2:  # モノクロ画像の場合（高さ, 幅）
        image_cv = np.expand_dims(image_cv, axis=-1)  # チャンネル次元を追加
    elif len(image_cv.shape) == 3:  # カラー画像の場合（高さ, 幅, チャンネル）
        if image_cv.shape[2] != 3:
            raise ValueError(f"Unexpected number of channels: {image_cv.shape[2]}")

    # 画像を(チャンネル, 高さ, 幅)に変換
    image = np.transpose(image_cv, (2, 0, 1)).astype(np.float32) / 255
    image = image[np.newaxis, :, :, :]
    image = torch.Tensor(image).to(device)

    # 3. 頭マップを作成
    headmap_np = image.sigmoid().cpu().detach().numpy()[0, 0]
    gray = (headmap_np * 255).astype(np.uint8)

    # 4. モード値を計算
    mode_result = stats.mode(gray.flatten(), keepdims=False)  # モードを計算
    mode_value = float(mode_result[0])  # スカラー値として扱うために float に変換
    print(f"Background mode value: {mode_value}")

    # 5. 二値化（モードをしきい値として使用）
    _, binary = cv2.threshold(gray, mode_value, 255, cv2.THRESH_BINARY)

    # 6. 二値画像をnp.uint8に変換
    binary = binary.astype(np.uint8)

    # 7. 輪郭検出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 8. 黒い背景の画像を作成
    height, width = image_cv.shape[:2]  # 元の画像と同じサイズの黒画像
    black_background = np.zeros((height, width, 3), dtype=np.uint8)

    # 9. 黒い背景に輪郭を描画
    cv2.drawContours(black_background, contours, -1, (255, 255, 255), 2)  # 白色で輪郭を描画

    # 10. 結果を保存（画像名にiを付けて保存）
    #cv2.imwrite(os.path.join(output_dir, f'contour_on_black_gray_{str(i).zfill(8)}.png'), gray)
    #cv2.imwrite(os.path.join(output_dir, f'contour_on_black_binary_{str(i).zfill(8)}.png'), binary)
    cv2.imwrite('testdata/outline.png', black_background)
