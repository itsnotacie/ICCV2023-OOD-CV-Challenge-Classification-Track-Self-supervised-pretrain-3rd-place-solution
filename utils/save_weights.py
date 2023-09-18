import json
import os
import shutil
import math
import re
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]

def save_weights(root_path, save_path):
    setname = "all"
    imgs_path = os.path.join(root_path, setname)
    class_list = sorted(os.listdir(imgs_path), key=natural_key)

    imgs_info = {}
    for clas in class_list:
        imgs_info[clas] = []
    
    train_num = 0
    for root, dirs, files in os.walk(imgs_path):
        class_name = root.split("/")[-1]
        for f in files:
            imgs_info[class_name].append(f)
            train_num += 1

    version = "v0"
    save_weights = {}
    weights = [math.pow(math.e, (train_num - len(v)) / train_num) for k, v in imgs_info.items()]
    
    if version == "v0":
        save_weights["cervicalCells"] = weights
    else:
        save_weights["cervicalCells"] = list(weights / np.sum(weights))

    json_name = os.path.join(save_path, "weights{}.json".format(version))
    with open(json_name, "w") as f:
        json.dump(save_weights, f, indent=2) 


def count_hw():
    hw_dict = {}
    hs, ws = [], []
    for root, dirs, files in os.walk(root_path):
        class_name = root.split("/")[-1]
        for f in tqdm(files):
            img_data = cv2.imread(os.path.join(root, f))
            h, w = img_data.shape[:-1]
            # print(h, w)
            hw_str = "{}-{}".format(h, w)
            if hw_str not in hw_dict: hw_dict[hw_str] = 0
            hw_dict[hw_str] += 1
            
            hs.append(h)
            ws.append(w)
    
    print(hw_dict)
    print(np.mean(hs), np.mean(ws))
    
def count_class_num(root_path):
    classes = ["ASC-H&HSIL", "ASC-US&LSIL", "NILM", "SCC&AdC"]
    for root, dirs, files in os.walk(root_path):
        class_name = root.split("/")[-1]
        if class_name in classes:
            print(root.split("/")[-2], class_name, len(files))

def load_img_info(root_path):
    import exifread
    setname = "traintest"
    imgs_path = os.path.join(root_path, setname)
    
    for root, dirs, files in os.walk(imgs_path):
        for f in files[:5]:
            f = open(os.path.join(root, f), "rb")
            tags = exifread.process_file(f)

            for k, v in tags.items():
                print(k, v)

if __name__ == "__main__":
    
    root_path = r"/data/guofeng/classification/CervicalCells/secondary"
    save_path = r"/data/guofeng/classification/CervicalCells/secondary"
    
    save_weights(root_path, save_path)
    # count_class_num(root_path)
    # load_img_info(root_path)
    
