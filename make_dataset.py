import  h5py
import  scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from matplotlib import cm as CM
from image import *
import cv2 as cv

# root is the path to dataset
root='./datasets'

train = os.path.join(root,'overall/train', 'images')
test = os.path.join(root,'overall/test', 'images')
path_sets = [train,test]


img_paths  = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.png')):
        img_paths.append(img_path)


for  img_path  in img_paths:

    img=cv.imread(img_path)
    h,w=img.shape[:2]
    k=np.zeros((h,w))
    labels=[]
    with open(img_path.replace('.png','.txt').replace('images','labels')) as f:
        for line in f:
            line=line.replace('\n','')
            point=line.split(' ')
            labels.append([int(float(point[0])),int(float(point[1]))])

    for label in labels:
        if label[1]<h and label[0]<w:
            k[label[1],label[0]]=1

    k=gaussian_filter(k,3)
    with h5py.File(img_path.replace('.png', '.h5').replace('images', 'labels'), 'w') as hf:
         hf['density'] = k

