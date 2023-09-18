import os
import shutil
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def train_val_split(img_paths, ratio_train=0.8, ratio_val=0.1, seed=42):
    assert int(ratio_train + ratio_val) == 1
    train_img, val_img = train_test_split(img_paths, test_size=1 - ratio_train, random_state=seed)
    print("NUMS of train:val = {}:{}".format(len(train_img), len(val_img)))
    return train_img, val_img


def copy_files2dst(imgs, gt_dict, ori_path, dst_path, setname):
    
    for img in imgs:
        class_id = gt_dict[img.split(".")[0]]  
        dst_img_path = os.path.join(dst_path, setname, class_id)
        if not os.path.exists(dst_img_path): os.makedirs(dst_img_path)

        shutil.copy(os.path.join(ori_path, img), os.path.join(dst_img_path, img))

def random_split(root_path):
    ori_imgs_path = os.path.join(root_path, "imgs")
    ori_label_path = os.path.join(root_path, "train.csv")
    class_name_path = os.path.join(root_path, "classes")
    dst_imgs_path = os.path.join(root_path, "primary")
        
    imgs = os.listdir(ori_imgs_path)
    csv_data = pd.read_csv(ori_label_path).values
    
    gt_dict = {}
    for csv_line in csv_data:
        line_data = csv_line[0].split(" ")
        img_name = line_data[1]
        class_id = line_data[2]
        gt_dict[img_name] = class_id
    # class_dict = {}
    # with open(class_name_path, "r", encoding="utf-8") as f:
    #     for l in f.readlines():
    #         line_data = l.strip().split(" ")
    #         class_dict[line_data[0]]
    #     lines = [) for l in f.readlines()]
    # print(csv_data, lines)
    
    train_img, val_img = train_val_split(imgs, 0.8, 0.2, seed=42)
    
    hs, ws = [], []
    for img in imgs:
        img_data = cv2.imread(os.path.join(ori_imgs_path, img))
        h, w = img_data.shape[:-1]
        hs.append(h)
        ws.append(w)
    
    print(np.mean(hs), np.mean(ws))
    # copy_files2dst(train_img, gt_dict, ori_imgs_path, dst_imgs_path, "train")
    # copy_files2dst(val_img, gt_dict, ori_imgs_path, dst_imgs_path, "val")


if __name__=="__main__":
    
    root_path = r"/data/guofeng/classification/LEDFlaw"
    
    random_split(root_path)