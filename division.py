import os
import cv2
import random
import shutil
import numpy as np
from tqdm import tqdm
from collections import defaultdict

'''
源文件夹下的图片和标签文件需要命名为images和labels，labels可以不用
记得修改image的扩展名:jpg  or  png
'''

'''将每个类别分成 train test valid'''

def main(src_dir, dst_dir, val_ratio=0.1, test_ratio=0.1):
    src_imgs_dir = os.path.join(src_dir, 'images')
    src_labels_dir = os.path.join(src_dir, 'labels')
    fids = [f for f in os.listdir(src_imgs_dir) if f.endswith('.jpg')]
    random.seed(6)
    random.shuffle(fids)
    total_num = len(fids)
    val_num = int(total_num * val_ratio)
    test_num = int(total_num * test_ratio)
    train_num = total_num - val_num - test_num

    train_set = fids[:train_num]
    val_set = fids[train_num: train_num + val_num]
    test_set = fids[train_num + val_num:]

    print('train num: {}  val num: {} test num: {}'.format(len(train_set), len(val_set), len(test_set)))

    out_train_dir = os.path.join(dst_dir, 'train')
    out_val_dir = os.path.join(dst_dir, 'val')

    os.makedirs(os.path.join(out_train_dir, 'images'), exist_ok=True)
    # os.makedirs(os.path.join(out_train_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(out_val_dir, 'images'), exist_ok=True)
    # os.makedirs(os.path.join(out_val_dir, 'labels'), exist_ok=True)

    for fid in tqdm(train_set):
        shutil.copy(os.path.join(src_imgs_dir, fid), os.path.join(out_train_dir, 'images', fid))
        # shutil.copy(os.path.join(src_labels_dir, fid), os.path.join(out_train_dir, 'labels', fid))
    for fid in tqdm(val_set):
        shutil.copy(os.path.join(src_imgs_dir, fid), os.path.join(out_val_dir, 'images', fid))
        # shutil.copy(os.path.join(src_labels_dir, fid), os.path.join(out_val_dir, 'labels', fid))
    if test_ratio:
        out_test_dir = os.path.join(dst_dir, 'test')
        os.makedirs(os.path.join(out_test_dir, 'images'), exist_ok=True)
        # os.makedirs(os.path.join(out_test_dir, 'labels'), exist_ok=True)
        for fid in tqdm(test_set):
            shutil.copy(os.path.join(src_imgs_dir, fid), os.path.join(out_test_dir, 'images', fid))
            # shutil.copy(os.path.join(src_labels_dir, fid), os.path.join(out_test_dir, 'labels', fid))
    print('done!')


if __name__ == '__main__':
    src_dir = r'C:\Users\Judy\Desktop\Zeng\classification_isic17\dataset\nevus'  # 源文件
    dst_dir = r'C:\Users\Judy\Desktop\Zeng\classification_isic17\dataset\nevus'  # 目标目录
    main(src_dir, dst_dir, val_ratio=0.1, test_ratio=0.1)  # 不需要test测试集就把test_ratio设为0
