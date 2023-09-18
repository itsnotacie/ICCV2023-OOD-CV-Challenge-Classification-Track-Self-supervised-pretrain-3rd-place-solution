import csv
import os
import glob
import shutil
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def gen_img(size=(1024, 512), color=(0, 0, 0)):
    # 创建一个新图片,定义好宽和高
    image = Image.new('RGB', size, color=color)
    return image

def draw_text(image, font_set):
    # # 创建画刷，用来写文字到图片上
    draw = ImageDraw.Draw(image)
    # 设置字体类型和大小
    font = ImageFont.truetype(font_set["type"], font_set["size"])
    # 根据位置、内容、颜色、字体来画进图片里,
    draw.text(font_set["location"], font_set["content"], font_set["color"], font=font)
    
def show_diff(csv_path, img_path):
    font_set = {
        "type": "/home/projects/TorchClassification/output/Arial.ttf",
        # "type": "/data/guofeng/others/yolov5/Arial.ttf",
        "size": 50,
        "content": "要写入的内容",
        "color": (255, 255, 255),
        "location": (122, 122),
    }
    
    # classes_list = ["upperLength", "clothesStyles", "hairStyles", "lowerLength", 
    #                 "lowerStyles", "shoesStyles", "towards", "upperColors", "lowerColors"]
    
    # classes_dict = {"upperLength": ["LongSleeve", "NoSleeve", "ShortSleeve"], # “长袖”、“无袖”、“短袖”
    #                 "clothesStyles": ["Solidcolor", "lattice", "multicolour"], # “纯色”、“格子”、“多色”
    #                 "hairStyles": ["Bald", "Long", "Short", "middle"], # “秃”、“长”、“短”、“中”
    #                 "lowerLength": ["Shorts", "Skirt", "Trousers"], # “短裤”、“裙子”、“裤子”
    #                 "lowerStyles": ["Solidcolor", "lattice", "multicolour"],
    #                 "shoesStyles": ["LeatherShoes", "Sandals", "Sneaker", "else"], # “皮鞋”、“凉鞋”、“运动鞋”、“其他”
    #                 "towards": ["back", "front", "left", "right"], # “后”、“前”、“左”、“右”
    # }
    
    # en2cn = {"upperLength": ["长袖", "无袖", "短袖"],
    #          "clothesStyles": ["纯色", "格子", "多色"],
    #          "hairStyles": ["秃", "长", "短", "中"],
    #          "lowerLength": ["短裤", "裙子", "裤子"],
    #          "lowerStyles": ["纯色", "格子", "多色"],
    #          "shoesStyles": ["皮鞋", "凉鞋", "运动鞋", "其他"],
    #          "towards": ["后", "前", "左", "右"]}
    
    save_path = os.path.join(csv_path, "imgsv8")
    if os.path.exists(save_path): shutil.rmtree(save_path)
    os.makedirs(save_path)
    
    models = [
        # "102919-convnext_large_384_in22ft1k-512#101516-convnext_large_384_in22ft1k-512",
        # "102332-convnext_large_384_in22ft1k-512#102522-convnext_large_384_in22ft1k-512#101415-convnext_large_384_in22ft1k-512",
        # "225239-convnext_large_384_in22ft1k-384#230230-convnext_large_384_in22ft1k-512",
        # "144406-convnext_large_384_in22ft1k-512#102332-convnext_large_384_in22ft1k-512",
        # "105636-convnext_large_384_in22ft1k-512#172346-convnext_large_384_in22ft1k-512",
        "102332-convnext_large_384_in22ft1k-512",
        "yiwei",
    ]
    print(models)
    csv_path0 = glob.glob(os.path.join(csv_path, models[0], "*B*"))[0]
    csv_path1 = glob.glob(os.path.join(csv_path, models[1], "*B*"))[0]
    
    csv_data0 = pd.read_csv(csv_path0).values[:, :8]
    csv_data1 = pd.read_csv(csv_path1).values[:, :8]
    
    for i, line0 in enumerate(csv_data0):
        line1 = csv_data1[i]
        diff_str = []
        img_name = line0[0]
        feats0 = line0[1:]
        feats1 = line1[1:]
        for fn in range(len(feats0)):
            if feats0[fn] != feats1[fn]:
                # print(fn ,classes_list[fn], classes_dict[classes_list[fn]], feats0[fn])
                # print(classes_dict[classes_list[fn]].index(feats0[fn]))
                # cn0 = en2cn[classes_list[fn]][classes_dict[classes_list[fn]].index(feats0[fn])]
                # cn1 = en2cn[classes_list[fn]][classes_dict[classes_list[fn]].index(feats1[fn])]
                diff_str.append("{}-{}".format(feats0[fn], feats1[fn]))
                # diff_str += "{}-{}".format(cn0, cn1)
        
        if len(diff_str) == 0: continue
        pil_img = Image.open(os.path.join(img_path, img_name))
        
        w, h = pil_img.size

        # size = 512
        # 首先创建一张全黑图像
        offset = 100
        target_image = gen_img(size=(w, h+offset))
        target_image.paste(pil_img, (0, offset, w, h+offset))
        
        for i in range(len(diff_str)):
            # 第一行数据的初始 Y 轴位置
            h_offset = 10 * (i + 1)
            # 第一个文字与图片的距离
            w_offset = 5
        
            font_set["location"] = (w_offset, h_offset)
            font_set["content"] = diff_str[i]
            font_set["size"] = 10
            draw_text(target_image, font_set)
        target_image.save(os.path.join(save_path, img_name))  
    

if __name__=="__main__":
    
    csv_path = r"/home/projects/TorchClassification/output/fusions/tta1"
    img_path = r"/data/guofeng/classification/person/secondary/testB"
    
    show_diff(csv_path, img_path)