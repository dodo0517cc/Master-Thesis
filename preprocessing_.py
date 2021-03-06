# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
import os
import cv2
import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.utils import model_zoo
import nibabel as nib
import nibabel.processing
# import skimage.io as io
# from skimage.transform import resize

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import sys
import time

def find_dir(path):
  for fd in os.listdir(path):
    full_path=os.path.join(path,fd)
    if os.path.isdir(full_path):
      # print('資料夾:',full_path)
      find_dir(full_path)
    else:
      # print('檔案:',full_path)
      data_path.append(full_path)

# !pip install SimpleITK

# import os
# import SimpleITK as sitk

"""# AD"""

data_path = []
AD_path = r'D:\Ubuntu\4class\LMCI'
SEED = 1
random.seed(SEED)
find_dir(AD_path)
AD_path_list = data_path
random.shuffle(AD_path_list)
min_p = 0.001
max_p = 0.999
voxel_size = [2, 2, 2]
for i in range(len(AD_path_list)):
  image_path = AD_path_list[i]
  img = nib.load(image_path) 
  img = nib.processing.resample_to_output(img, voxel_size)
  img = np.array(img.dataobj)
  img = np.float32(img)
#   print(img.shape)

  imgPixel = img[img >= 0]
  imgPixel.sort()
  index = int(round(len(imgPixel) - 1) * min_p + 0.5)
  if index < 0:
    index = 0
  if index > (len(imgPixel) - 1):
    index = len(imgPixel) - 1
  value_min = imgPixel[index]

  index = int(round(len(imgPixel) - 1) * max_p + 0.5)
  if index < 0:
    index = 0
  if index > (len(imgPixel) - 1):
    index = len(imgPixel) - 1
  value_max = imgPixel[index]

  mean = (value_max + value_min) / 2.0
  stddev = (value_max - value_min) / 2.0
  img = (img - mean) / stddev
  img[img < -1] = -1.0
  img[img > 1] = 1.0


  transform1 = transforms.Compose([
      transforms.ToTensor()
      ])
  img = transform1(img)
  img = img.permute((0,2,1))
  img = img.unsqueeze(0)
  img = aug_pad(img)

  if i == 0:
    image = img
    continue
  image = torch.cat((image, img), 0)
  if i%100 == 0:
    print(i)

#10:90,5:115,10:110
plt.imshow(image[3,60,2:120,10:110],cmap='gray')

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage import measure

def plot_3d(image, threshold=-0.2): 
#     p = image
    p = image.permute(1,2,0)
    p = np.array(p)
    verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
#     face_color = [0.5, 0.5, 1]
#     mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

plot_3d(image[3])

plt.imshow(image[1,55,:,:],cmap='gray')

image.shape



np.save(r"D:\Ubuntu\4class\AD_seg_voxel.npy", image)

"""# CN"""

data_path = []
CN_path = 'D:\\Ubuntu\\seg_CN'
find_dir(CN_path)
CN_path_list = data_path
random.shuffle(CN_path_list)

min_p = 0.001
max_p = 0.999
voxel_size = [2, 2, 2]
for i in range(len(CN_path_list)):
  image_path = CN_path_list[i]
  img = nib.load(image_path)
  img = nib.processing.resample_to_output(img, voxel_size)
  img = np.array(img.dataobj)
  img = np.float32(img)

  imgPixel = img[img >= 0]
  imgPixel.sort()
  index = int(round(len(imgPixel) - 1) * min_p + 0.5)
  if index < 0:
    index = 0
  if index > (len(imgPixel) - 1):
    index = len(imgPixel) - 1
  value_min = imgPixel[index]

  index = int(round(len(imgPixel) - 1) * max_p + 0.5)
  if index < 0:
    index = 0
  if index > (len(imgPixel) - 1):
    index = len(imgPixel) - 1
  value_max = imgPixel[index]

  mean = (value_max + value_min) / 2.0
  stddev = (value_max - value_min) / 2.0
  img = (img - mean) / stddev
  img[img < -1] = -1.0
  img[img > 1] = 1.0


  transform1 = transforms.Compose([
      transforms.ToTensor()
      ])
  img = transform1(img)
  img = img.permute((0,2,1))
  img = img.unsqueeze(0)
  img = aug_pad(img)
  # img = resize(img, (1, 108, 128, 128))
  # img = torch.from_numpy(img)

  if i == 0:
    image = img
    continue
  image = torch.cat((image, img), 0)

  if i%10 == 0:
    print(i)

image.shape

# np.save("/content/drive/MyDrive/Lab/data_all/CN_seg_voxel.npy", image)
np.save("D:\\Ubuntu\\4class\\CN_seg_voxel.npy", image)

"""## EMCI"""

data_path = []
MCI_path = 'D:\\Ubuntu\\seg_EMCI'
find_dir(MCI_path)
MCI_path_list = data_path
random.shuffle(MCI_path_list)

print(len(data_path))

min_p = 0.001
max_p = 0.999
voxel_size = [2, 2, 2]
for i in range(400):
  image_path = MCI_path_list[i]
  img = nib.load(image_path)
  img = nib.processing.resample_to_output(img, voxel_size)
  img = np.array(img.dataobj)
  img = np.float32(img)

  imgPixel = img[img >= 0]
  imgPixel.sort()
  index = int(round(len(imgPixel) - 1) * min_p + 0.5)
  if index < 0:
    index = 0
  if index > (len(imgPixel) - 1):
    index = len(imgPixel) - 1
  value_min = imgPixel[index]

  index = int(round(len(imgPixel) - 1) * max_p + 0.5)
  if index < 0:
    index = 0
  if index > (len(imgPixel) - 1):
    index = len(imgPixel) - 1
  value_max = imgPixel[index]

  mean = (value_max + value_min) / 2.0
  stddev = (value_max - value_min) / 2.0
  img = (img - mean) / stddev
  img[img < -1] = -1.0
  img[img > 1] = 1.0


  transform1 = transforms.Compose([
      transforms.ToTensor()
      ])
  img = transform1(img)
  img = img.permute((0,2,1))
  img = img.unsqueeze(0)
  img = aug_pad(img)
  # img = resize(img, (1, 108, 128, 128))
  # img = torch.from_numpy(img)

  if i == 0:
    image = img
    continue
  image = torch.cat((image, img), 0)

  if i%10 == 0:
    print(i)

image.shape

np.save("D:\\Ubuntu\\4class\\MCI_seg_voxel_try.npy", image)

