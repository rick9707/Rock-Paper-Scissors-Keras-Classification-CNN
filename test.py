# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from PIL import Image

import os, sys
import numpy as np
from glob import glob
import pandas as pd


def resize_images(img_path, target_size):

    if not os.path.isdir("./test3"):
        os.mkdir("./test3")
    
    resized_img_path = "./test3"
    img_list = glob(img_path + "/*")
    sorted_img_list = sorted(img_list, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    for i, img in enumerate(sorted_img_list):
        original_img = Image.open(img).convert('RGB')
        resized_img = original_img.resize(target_size, Image.ANTIALIAS)
        img_name = img.split("/")[-1].split('.')[0]
        resized_img.save('./test3/'+img_name+".png")



def load_test(img_path, img_size): 
    img_list = glob('./test3/*')
    number_of_data = len(img_list)
    color=3
    imgs = np.zeros(number_of_data * img_size * img_size * color,
                    dtype = np.int32)\
        .reshape(number_of_data, img_size, img_size, color)
    labels=[]
    idx=0

    img_list = glob('./test3/*')
    sorted_img_list = sorted(img_list, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    print(sorted_img_list)
    for i, image in enumerate(sorted_img_list):
        print(i,image)
        img = np.array(Image.open(image),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels.append(image.split("/")[-1].split('.')[0])
        idx=idx+1


    return imgs, labels

if __name__ == '__main__':
    image_dir_path = sys.argv[-1]
    model = keras.models.load_model('./model')
    resize_images(image_dir_path, (128, 128))
    (x, y)=load_test("./test3", 128)
    predictions = model.predict(x)
    print("y_test = ", y[:10])
    print("predic = ", np.argmax(predictions[:10], axis=1))
    result = []
    idx_for_adjust = {0: 2, 1: 0, 2: 1}
    idx_for_win = {0: 2, 1: 0, 2: 1}
    for i in range(len(predictions)):
        result.append([y[i], idx_for_adjust[int(np.argmax(predictions[i]))]])
    df = pd.DataFrame(result)
    print(df)
    df.to_csv("./output.txt", sep="\t", index=False, header=False)