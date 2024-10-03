import time
import torch
import segmentation_models_pytorch as smp

import numpy as np
import cv2
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='',  # プログラム名
        usage='',  # プログラムの利用方法
        add_help=True,  # -h/–help オプションの追加
    )
    parser.add_argument('--model', type=str, default='model_010.pth')
    parser.add_argument('--input', type=str, default='val/image_00000000.png')
    parser.add_argument('--label', type=str, default='')
    parser.add_argument('--output', type=str, default='output.png')
    args = parser.parse_args()
    alpha = 0.5
    model_path = args.model
    image_path = args.input
    output_path = args.output

    model = torch.load(model_path)
    model.eval()

    image_cv = cv2.imread(image_path)
    image = np.transpose(image_cv, (2,0,1)).astype(np.float32)/255
    image = image[np.newaxis,:,:,:]
    image = torch.Tensor(image).to("cuda")

    pred = model(image)
    headmap_np = pred.sigmoid().cpu().detach().numpy()[0,0]
    headmap_gray = (headmap_np*255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(headmap_gray, cv2.COLORMAP_JET)
    blend = cv2.addWeighted(image_cv, alpha, heatmap_color, 1-alpha, 0)
    if args.label == '':
        show_img = cv2.hconcat([image_cv, blend])
    else :
        label_img = cv2.imread(args.label, cv2.IMREAD_COLOR)
        show_img = cv2.hconcat([image_cv, label_img, blend])
    cv2.imwrite(output_path, show_img)
