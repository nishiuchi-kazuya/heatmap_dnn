import os
import json
import cv2
import numpy as np

if __name__ == '__main__':
    image_path_roots = ['/dataset/coco/train2017', '/dataset/coco/val2017']
    load_annotation_files = ['/dataset/coco/annotations/person_keypoints_train2017.json', '/dataset/coco/annotations/person_keypoints_val2017.json']
    
    for ds_type, image_path_root, load_annotation_file in zip (['train', 'val'], image_path_roots, load_annotation_files):
        with open(load_annotation_file) as f:
            coco_dict = json.load(f)
        ds_list = []
        cnt = 0
        ds_max = 25000 if ds_type == 'train' else 5000
        # ds_max = 10 if ds_type == 'train' else 10
        for i, anno in enumerate(coco_dict["annotations"]):
            if anno['iscrowd'] == 1 or anno['num_keypoints'] == 0:
                continue
            image_id = anno['image_id']
            image_dat = [img for img in coco_dict['images'] if img['id']==image_id][0]
            image_path = os.path.join(image_path_root, image_dat['file_name'])
            image = cv2.imread(image_path)
            keypoint_map = np.zeros(image.shape[0:2], dtype=np.uint8)
            keypoint_data = np.array(anno['keypoints']).reshape((-1,3))
            if (keypoint_data[[6,7,8,9],2]==2).all():
                for k_id in [6,7,8,9]:
                    k_x = keypoint_data[k_id][0]
                    k_y = keypoint_data[k_id][1] 
                    keypoint_map[k_y-1:k_y+2, k_x-1:k_x+2] = 255
            else :
                continue
            bbox = np.array(anno['bbox'])
            xs = int(bbox[0])
            ys = int(bbox[1])
            xe = int(bbox[0]+bbox[2])
            ye = int(bbox[1]+bbox[3])
            crop_image = image[ys:ye, xs:xe]
            crop_keypoint_map = keypoint_map[ys:ye, xs:xe]
            crop_image_resize = cv2.resize(crop_image, (256,256))
            crop_keypoint_map_resize = cv2.resize(crop_keypoint_map, (256,256), cv2.INTER_NEAREST)
            save_image_file = "{0}/image_{1:08d}.png".format(ds_type, i)
            save_label_file = "{0}/label_{1:08d}.png".format(ds_type, i)
            cv2.imwrite(save_image_file, crop_image_resize)
            cv2.imwrite(save_label_file, crop_keypoint_map_resize)
            ds_list.append((save_image_file, save_label_file))
            if False :
                debug_image = cv2.hconcat([crop_image_resize, cv2.cvtColor(crop_keypoint_map_resize, cv2.COLOR_GRAY2BGR)])
                cv2.imwrite("debug_{0:08d}.png".format(i), debug_image)
                if i > 100:
                    exit()
            cnt += 1
            if cnt >= ds_max:
                break
        with open('{}.csv'.format(ds_type), 'w') as f:
            for d in ds_list:
                print(*d, sep=',', file=f)
