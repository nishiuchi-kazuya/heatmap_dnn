import os
import time
import torch
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import datetime
import numpy as np
import pandas as pd
import cv2
import argparse
from setuptools._distutils.util import strtobool

# heatmap用データセット
class HeatmapDatasets(torch.utils.data.Dataset):
    ##__〇〇__はダンダーメソッドといい、メソッドを呼ぶ、
    # というよりはかってにやってくれるものにつく
    def __init__(self, csv_path, input_size=None):
        #csvファイルの読み込み、ディレクトリと画像サイズを設定
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path, header=None)
        self.input_size = input_size
        self.dir = os.path.abspath(os.path.dirname(csv_path))

    def __len__(self):
        #データセットのサンプル数を返す
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        #指定したインデックスの画像とラベルを取得
        #画像へのパスを変数に入れてアクセスしているんじゃなくてcsvファイルを見て、そこに書いてある画像へアクセスしている
        image_path = os.path.join(self.dir, self.df[0][idx])
        
        
        #入力に対してエンドエフェクタの正解とジョイントの正解がある
        label_end_path = os.path.join(self.dir, self.df[1][idx])
        label_joint_path = os.path.join(self.dir, self.df[2][idx])
        
        image = cv2.imread(image_path)
        label_end = cv2.imread(label_end_path, cv2.IMREAD_GRAYSCALE)
        label_joint = cv2.imread(label_joint_path, cv2.IMREAD_GRAYSCALE)
        if self.input_size is not None:
            #画像とラベルを指定サイズにリサイズ
            image = cv2.resize(image, self.input_size)
            label_end = cv2.resize(label_end, self.input_size, cv2.INTER_NEAREST)
            label_joint = cv2.resize(label_joint, self.input_size, cv2.INTER_NEAREST)
             

        #ここからわからん
        image = image.astype(np.float32)/255#画像を0~1の範囲に正規化
        image = np.transpose(image, (2,0,1))#画像のチャネル順序を(高さ,幅,チャネル)から(チャネル,高さ,幅)へ
        label_end[label_end!=0]=1#labelの非ゼロを1にして二値化
        label_joint[label_joint!=0]=1#labelの非ゼロを1にして二値化
        #2つのラベル画像 (label1 と label2) をまとめて1つの多次元配列にスタックし、整数型に変換する処理
        #labels=(2,512,512)
        labels = np.stack((label_end, label_joint), axis=0).astype(np.int64)
        return image, labels

# 精度を計算する関数を追加
def calculate_accuracy(predicted, ground_truth):
    """
    Accuracy calculation for binary masks
    """
    predicted = predicted.sigmoid() >= 0.5  # 確率を閾値0.5でマスクに変換
    correct = (predicted == ground_truth).sum().item()  # 正しい予測の数
    total = ground_truth.numel()  # 全要素数
    return correct / total
if __name__ == '__main__':
    #コマンドライン引数の設定
    parser = argparse.ArgumentParser(
        prog='',  # プログラム名
        usage='',  # プログラムの利用方法
        add_help=True,  # -h/–help オプションの追加
    )
    #parser.add_argument('--traincsv', type=str, default='joint_dataset/outline_train.csv')
    #parser.add_argument('--valcsv', type=str, default='joint_dataset/outline_val.csv')
    #parser.add_argument('--traincsv', type=str, default='joint_dataset/image+outline_train.csv')
    #parser.add_argument('--valcsv', type=str, default='joint_dataset/image+outline_val.csv')
    parser.add_argument('--traincsv', type=str, default='joint_dataset/image_train.csv')
    parser.add_argument('--valcsv', type=str, default='joint_dataset/image_val.csv')
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--inputsize', type=int, nargs='*', default=None)
    parser.add_argument('--useamp', type=strtobool, default=0)
    parser.add_argument('--arch', type=str, default='Unet')
    parser.add_argument('--encoder', type=str, default='resnet18')
    args = parser.parse_args()

    #入力サイズの処理(512*512)
    inputsize = args.inputsize
    if type(inputsize) is list:
        if len(inputsize) == 0:
            inputsize = None
        else :
            if len(inputsize) == 1:
                inputsize = inputsize+inputsize
            elif len(inputsize) > 2:
                inputsize = inputsize[:2]
            inputsize = tuple(inputsize)

    #デバイスと出力ディレクトリの設定
    device = args.device
    date_string = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')#時間を取得してモデルの名前にする
    
    outputdir = './image+outline_model/output-{}'.format(date_string) if args.output == '' else args.output
    #outputdir = './outline_model/output-{}'.format(date_string) if args.output == '' else args.output
    use_amp = args.useamp == 1
    scaler = torch.amp.GradScaler(enabled=use_amp, init_scale=4096)#amp(16bitの浮動小数点)により学習を高速化しつつメモリ使用量を削減。スケーリングの初期値を4096にする。初期値が大きいほど勾配のアンダーフローを避けれる。

    # モデル定義
    model = smp.create_model(
        arch = args.arch,
        encoder_name = args.encoder,
        encoder_weights="imagenet",
        #変更点
        classes=2#2チャネルの出力
    )
    model.to(device)
    
    # データローダ準備
    abs_traincsv_path = os.path.abspath(args.traincsv)
    abs_valcsv_path = os.path.abspath(args.valcsv)

    #ここがインスタンスを生成しているところなので__initにあるdirとかはここ参照してね
    #self.dir=/home/irsl/heatmap_dnn/
    #csvは/home/irsl/datasetにあって、datasetはuserdirにまうんとしてるから
    #trainset = HeatmapDatasets(args.traincsv, input_size = inputsize) 元の
    trainset = HeatmapDatasets(os.path.join('dataset', args.traincsv), input_size = inputsize)#俺のファイル階層にあてはめたもの
    #valset = HeatmapDatasets(args.valcsv, input_size = inputsize)元の
    valset = HeatmapDatasets(os.path.join('dataset', args.valcsv), input_size = inputsize)#俺のファイル階層にあてはめたもの
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size = args.batchsize, shuffle = True, num_workers = 8)
    val_dataloader = torch.utils.data.DataLoader(valset, batch_size = 1, shuffle = False, num_workers = 2)
    #print(len(trainset))#データセットの数

    # 評価関数，最適化関数定義
    loss_func = smp.losses.FocalLoss(mode='multilabel')
    metric_mae_func = torch.nn.L1Loss()
    metric_mse_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=args.lr)])

    # 出力先ディレクトリ作成
    os.makedirs(outputdir, exist_ok=True)

    # loggerロガー準備
    writer = SummaryWriter(log_dir=outputdir)
    itr_num = 0

    #エポック(訓練データをすべて使い切って一周した時を1とする訓練データを何回用いたかを表す数)
    # のループ（エポック多いと過学習おきる）
    with tqdm(range(args.epoch)) as pbar_epoch:
        for ep in pbar_epoch:
            pbar_epoch.set_description("[Epoch %d]" % (ep))
            #トレーニングループ
            with tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False) as pbar_itr:
                for i, (images, gt_masks) in pbar_itr:
                    images = images.to(device)
                    gt_masks = gt_masks.to(device)

                    optimizer.zero_grad()
                    with torch.autocast(device_type=device, enabled=use_amp):
                        predicted_mask = model(images)
                        loss = loss_func(predicted_mask, gt_masks)

                    # 勾配の計算と更新
                    scaler.scale(loss).backward()

                    # 勾配ノルムの計算
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()

                    # 精度の計算
                    accuracy = calculate_accuracy(predicted_mask, gt_masks)

                    # 現在の学習率を取得
                    current_lr = optimizer.param_groups[0]['lr']

                    # TensorBoardにログを記録
                    writer.add_scalar("training_loss", loss.cpu().item(), itr_num)
                    writer.add_scalar("training_accuracy", accuracy, itr_num)
                    writer.add_scalar("gradient_norm", grad_norm, itr_num)
                    writer.add_scalar("learning_rate", current_lr, itr_num)

                    
                    #サンプル数の記録
                    if itr_num % 100 == 0:
                        predicted_mask_work = predicted_mask.sigmoid()
                        predicted_mask_work2 = predicted_mask_work[:,1:2,:,:]
                        predicted_mask_show = torch.cat([predicted_mask_work,predicted_mask_work2], dim=1)
                        gt_masks_work = gt_masks #torch.unsqueeze(gt_masks,1)
                        gt_masks_work2 = gt_masks_work[:,1:2,:,:]
                        gt_masks_show = torch.cat([gt_masks_work, gt_masks_work2], dim=1)
                        img_sample = (torch.cat([images, gt_masks_show, predicted_mask_show], dim=3)*255).to(torch.uint8)
                        writer.add_images("train_example", img_sample, itr_num, dataformats='NCHW')
                    itr_num += 1
                    if False: # for debug
                        save_images = []
                        for j in range(3):
                            alpha = 0.5
                            input_image = (images.cpu().numpy()[j]*255).astype(np.uint8).transpose((1,2,0))
                            label_heatmap = (gt_masks.cpu().numpy()[j]*255).astype(np.uint8)
                            pred_heatmap = (predicted_mask.cpu().detach().sigmoid().numpy()[j][0]*255).astype(np.uint8)
                            pred_heatmap_color = cv2.applyColorMap(pred_heatmap, cv2.COLORMAP_JET)
                            blend = cv2.addWeighted(input_image, alpha, pred_heatmap_color, 1-alpha, 0)
                            # save_image = cv2.hconcat([input_image, cv2.cvtColor(label_heatmap, cv2.COLOR_GRAY2BGR), cv2.cvtColor(pred_heatmap, cv2.COLOR_GRAY2BGR)])
                            # save_image = cv2.hconcat([input_image, cv2.cvtColor(label_heatmap, cv2.COLOR_GRAY2BGR), pred_heatmap_color])
                            save_image = cv2.hconcat([input_image, cv2.cvtColor(label_heatmap, cv2.COLOR_GRAY2BGR), cv2.cvtColor(pred_heatmap, cv2.COLOR_GRAY2BGR), blend])
                            save_images.append(save_image)
                        save_images2 = cv2.vconcat(save_images)
                        cv2.imwrite("test.png", save_images2)
                        # time.sleep(0.05)

            # val datasetを用いてloss, metricを確認
            #検証ループ
            model.eval()
            loss_list = []
            metric_mae_list = []
            metric_mse_list = []
            validation_accuracies = []
            for i, (images, gt_masks) in enumerate(val_dataloader):
                images = images.to(device)
                gt_masks = gt_masks.to(device)
                with torch.no_grad():
                    predicted_mask = model(images)
                    loss = loss_func(predicted_mask, gt_masks)
                    accuracy = calculate_accuracy(predicted_mask, gt_masks)  # 精度を計算
                    predicted_mask_sigmoid = predicted_mask.sigmoid()
                    metric_mae = metric_mae_func(torch.squeeze(predicted_mask_sigmoid, dim=1), gt_masks)
                    metric_mse = metric_mse_func(torch.squeeze(predicted_mask_sigmoid, dim=1), gt_masks)
                loss_list.append(loss.item())
                metric_mae_list.append(metric_mae.item())
                metric_mse_list.append(metric_mse.item())
                validation_accuracies.append(accuracy)  # 検証精度を記録

            # TensorBoardに検証ログを記録
            writer.add_scalar("validation/loss", np.mean(loss_list), ep+1)
            writer.add_scalar("validation/metric/mae", np.mean(metric_mae_list), ep+1)
            writer.add_scalar("validation/metric/mse", np.mean(metric_mse_list), ep+1)
            writer.add_scalar("validation/accuracy", np.mean(validation_accuracies), ep+1)

            model.train()

            #torch.save(model, os.path.join(outputdir, "./outline_model/model_{0:03d}.pth".format(ep+1)))
            torch.save(model, os.path.join(outputdir, "model_{0:03d}.pth".format(ep+1)))
    # end epoch loop
    writer.close()