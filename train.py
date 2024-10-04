
import os
import time
import torch
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter

import datetime
import numpy as np
import pandas as pd
import cv2
import argparse

# headmap用データセット
class HeadmapDatasets(torch.utils.data.Dataset):
    def __init__(self, csv_path, csv_dir, input_size=None):
        self.df = pd.read_csv(csv_path, header=None)
        self.dir = csv_dir
        self.input_size = input_size
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, idx):
        image_path = os.path.join(self.dir, self.df[0][idx])
        label_path = os.path.join(self.dir, self.df[1][idx])
        image = cv2.imread(image_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if self.input_size is not None:
            image = cv2.resize(image, self.input_size)
            label = cv2.resize(label, self.input_size, cv2.INTER_NEAREST)
        image = image.astype(np.float32)/255
        image = np.transpose(image, (2,0,1))
        label[label!=0]=1
        label = label.astype(np.int64)
        return image, label

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='',  # プログラム名
        usage='',  # プログラムの利用方法
        add_help=True,  # -h/–help オプションの追加
    )
    parser.add_argument('--traincsv', type=str, default='train.csv')
    parser.add_argument('--valcsv', type=str, default='val.csv')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    device = args.device
    date_string = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    outputdir = 'output-{}'.format(date_string) if args.output == '' else args.output

    # モデル定義
    model = smp.Unet (
        encoder_name="resnet18", 
        encoder_weights="imagenet", 
        classes=1, 
    )
    model.to(device)
    
    # データローダ準備
    abs_traincsv_path = os.path.abspath(args.traincsv)
    abs_valcsv_path = os.path.abspath(args.valcsv)
    trainset = HeadmapDatasets(args.traincsv, os.path.dirname(abs_traincsv_path))
    valset = HeadmapDatasets(args.valcsv, os.path.dirname(abs_valcsv_path))
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size = args.batchsize, shuffle = True, num_workers = 8)
    val_dataloader = torch.utils.data.DataLoader(valset, batch_size = 1, shuffle = False, num_workers = 2)

    # 評価関数，最適化関数定義
    loss_func = smp.losses.FocalLoss(mode='binary')
    metric_func = torch.nn.L1Loss()
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=5e-4)])

    # 出力先ディレクトリ作成
    os.makedirs(outputdir, exist_ok=False)

    # loggerロガー準備
    writer = SummaryWriter(log_dir=outputdir)
    itr_num = 0

    for ep in range(args.epoch):
        # training 
        for i, (images, gt_masks) in enumerate(train_dataloader):
            images = images.to(device)
            gt_masks = gt_masks.to(device)
            optimizer.zero_grad()
            predicted_mask = model(images)
            loss = loss_func(predicted_mask, gt_masks)
            loss.backward()
            optimizer.step()
            # print(ep, i, loss.cpu().item())
            writer.add_scalar("training_loss", loss.cpu().item(), itr_num)
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
        model.eval()
        loss_list = []
        metric_list = []
        for i, (images, gt_masks) in enumerate(val_dataloader):
            images = images.to(device)
            gt_masks = gt_masks.to(device)
            with torch.no_grad():
                predicted_mask = model(images)
                loss = loss_func(predicted_mask, gt_masks)
                metric = metric_func(torch.squeeze(predicted_mask.sigmoid(), dim=1), gt_masks)
            loss_list.append(loss.item())
            metric_list.append(metric.item())
        # print(ep+1, np.mean(loss_list), np.mean(metric_list))
        
        writer.add_scalar("validation/loss", np.mean(loss_list), ep+1)
        writer.add_scalar("validation/metric", np.mean(metric_list), ep+1)
        model.train()

        torch.save(model, os.path.join(outputdir, "./model_{0:03d}.pth".format(ep+1)))
    # end epoch loop
    writer.close()