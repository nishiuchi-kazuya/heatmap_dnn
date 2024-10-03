
import time
import torch
import segmentation_models_pytorch as smp

import numpy as np
import pandas as pd
import cv2
import argparse

# headmap用データセット
class HeadmapDatasets(torch.utils.data.Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, header=None)
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, idx):
        image_path = self.df[0][idx]
        label_path = self.df[1][idx]
        image = cv2.imread(image_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
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
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--output', type=str, default='output.png')

    DEVICE = "cuda"

    # モデル定義
    model = smp.Unet (
        encoder_name="resnet18", 
        encoder_weights="imagenet", 
        classes=1, 
    )
    model.to(DEVICE)

    trainset = HeadmapDatasets(args.traincsv)
    # valset = HeadmapDatasets('val.csv')
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size = args.batchsize, shuffle = True, num_workers = 8)
    # val_dataloader = torch.utils.data.DataLoader(valset, batch_size = 1, shuffle = False, num_workers = 2)

    loss_func = smp.losses.FocalLoss(mode='binary')
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=5e-4)])

    for ep in range(args.epoch):
        for i, (images, gt_masks) in enumerate(train_dataloader):
            images = images.to(DEVICE)
            gt_masks = gt_masks.to(DEVICE)
            optimizer.zero_grad()
            predicted_mask = model(images)
            loss = loss_func(predicted_mask, gt_masks)
            loss.backward()
            optimizer.step()
            print(ep, i, loss.cpu().item())
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
        torch.save(model, "./model_{0:03d}.pth".format(ep+1))