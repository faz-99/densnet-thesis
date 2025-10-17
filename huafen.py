import glob
import os
import shutil

adenosis=glob.glob("/home/wu/Jia/steel_cls/after-data/1/*")
filroadenoma=glob.glob("/home/wu/Jia/steel_cls/after-data/2/*")
pt=glob.glob("/home/wu/Jia/steel_cls/after-data/3/*")
ta=glob.glob("/home/wu/Jia/steel_cls/after-data/4/*")
dc=glob.glob("/home/wu/Jia/steel_cls/after-data/5/*")
lc=glob.glob("/home/wu/Jia/steel_cls/after-data/6/*")
mc=glob.glob("/home/wu/Jia/steel_cls/after-data/7/*")
#pc=glob.glob("/home/wu/Jia/steel_cls/data/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/malignant/SOB/papillary_carcinoma/*/40X/*.png")

for i in adenosis:
     path="/home/wu/Jia/steel_cls/steel_data/1"
     shutil.copy(i,path)

for i in filroadenoma:
     path="/home/wu/Jia/steel_cls/steel_data/2"
     shutil.copy(i,path)


for i in pt:
     path="/home/wu/Jia/steel_cls/steel_data/3"
     shutil.copy(i,path)

for i in ta:
     path="/home/wu/Jia/steel_cls/steel_data/4"
     shutil.copy(i,path)


for i in dc:
     path="/home/wu/Jia/steel_cls/steel_data/5"
     shutil.copy(i,path)

for i in lc:
     path="/home/wu/Jia/steel_cls/steel_data/6"
     shutil.copy(i,path)

for i in mc:
     path="/home/wu/Jia/steel_cls/steel_data/7"
     shutil.copy(i,path)


