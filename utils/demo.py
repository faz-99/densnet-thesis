import os
import shutil

### 数据集根目录
root_dir = '/home/wu/Jia/steel_cls/NEU-CLS'

### 数据集转移目录
shutil_dir = '/home/wu/Jia/steel_cls/dataset'

all_images = os.listdir(root_dir)   #读取所有文件

images_classes= ['Cr', 'In', 'Pa', 'PS', 'RS', 'Sc']

for img in all_images:
    img_shutil_dir = os.path.join(shutil_dir, str(images_classes.index(img[0:2])))
    if not os.path.isdir(img_shutil_dir):
        os.mkdir(img_shutil_dir)
    shutil.copyfile(os.path.join(root_dir, img), os.path.join(img_shutil_dir, img))