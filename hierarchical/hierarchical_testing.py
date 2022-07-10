#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install torch
# !pip install torchvision


# In[2]:


# !pip install packaging


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
import os
import sys
#import cv2
import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.utils import model_zoo
import nibabel as nib
# import skimage.io as io
# from skimage.transform import resize
from sklearn.metrics import confusion_matrix
import seaborn as sn


# In[4]:


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# In[5]:


os.chdir('/home/u9285752')


# # Load Data

# In[6]:


CN_image= np.load("./CN_seg_voxel.npy")
CN_image = CN_image[:,10:90,5:115,10:110]
CN_image = torch.from_numpy(CN_image)
CN_image.shape


# In[7]:


EMCI_image= np.load("./EMCI_seg_voxel.npy")
EMCI_image = EMCI_image[:,10:90,5:115,10:110]
EMCI_image = torch.from_numpy(EMCI_image)
EMCI_image.shape


# In[8]:


LMCI_image= np.load("./LMCI_seg_voxel.npy")
LMCI_image = LMCI_image[:,10:90,5:115,10:110]
LMCI_image = torch.from_numpy(LMCI_image)
LMCI_image.shape


# In[9]:


AD_image= np.load("./AD_seg_voxel.npy")
AD_image = AD_image[:,10:90,5:115,10:110]
AD_image = torch.from_numpy(AD_image)
AD_image.shape


# # Split Train & Test 

# In[12]:


idx = round(CN_image.shape[0]*0.8*0.8)
idx2 = round(CN_image.shape[0]*0.8)
idx3 = round(CN_image.shape[0]*0.9)
CN_train_image = CN_image[:idx]
print(CN_train_image.shape)

CN_val_image = CN_image[idx:idx2]
print(CN_val_image.shape)

CN_test_image = CN_image[idx2:]
print(CN_test_image.shape)


# In[13]:


idx = round(EMCI_image.shape[0]*0.8*0.8)
idx2 = round(EMCI_image.shape[0]*0.8)
idx3 = round(EMCI_image.shape[0]*0.9)
EMCI_train_image = EMCI_image[:idx]
print(EMCI_train_image.shape)

EMCI_val_image = EMCI_image[idx:idx2]
print(EMCI_val_image.shape)

EMCI_test_image = EMCI_image[idx2:]
print(EMCI_test_image.shape)


# In[14]:


idx = round(LMCI_image.shape[0]*0.8*0.8)
idx2 = round(LMCI_image.shape[0]*0.8)
idx3 = round(LMCI_image.shape[0]*0.9)
LMCI_train_image = LMCI_image[:idx]
print(LMCI_train_image.shape)

LMCI_val_image = LMCI_image[idx:idx2]
print(LMCI_val_image.shape)

LMCI_test_image = LMCI_image[idx2:]
print(LMCI_test_image.shape)


# In[15]:


idx = round(AD_image.shape[0]*0.8*0.8)
idx2 = round(AD_image.shape[0]*0.8)
idx3 = round(AD_image.shape[0]*0.9)
AD_train_image = AD_image[:idx]
print(AD_train_image.shape)

AD_val_image = AD_image[idx:idx2]
print(AD_val_image.shape)

AD_test_image = AD_image[idx2:]
print(AD_test_image.shape)


# In[17]:


del(AD_image)
del(CN_image)
del(EMCI_image)
del(LMCI_image)


# # Split data

# In[58]:


train_image = torch.cat((CN_train_image, EMCI_train_image, LMCI_train_image, AD_train_image), 0)
print(train_image.shape)

CN_train_label = torch.zeros(CN_train_image.shape[0])
EMCI_train_label = torch.ones(EMCI_train_image.shape[0])
LMCI_train_label = torch.ones(LMCI_train_image.shape[0])*2
AD_train_label = torch.ones(AD_train_image.shape[0])*3
train_label = torch.cat((CN_train_label, EMCI_train_label, LMCI_train_label, AD_train_label))
print(train_label.shape)


# In[59]:


val_image = torch.cat((CN_val_image, EMCI_val_image, LMCI_val_image, AD_val_image), 0)
print(val_image.shape)

CN_val_label = torch.zeros(CN_val_image.shape[0])
EMCI_val_label = torch.ones(EMCI_val_image.shape[0])
LMCI_val_label = torch.ones(LMCI_val_image.shape[0])*2
AD_val_label = torch.ones(AD_val_image.shape[0])*3

val_label = torch.cat((CN_val_label, EMCI_val_label, LMCI_val_label, AD_val_label))
print(val_label.shape)


# In[60]:


test_image = torch.cat((CN_test_image, EMCI_test_image, LMCI_test_image, AD_test_image), 0)
print(test_image.shape)


CN_test_label = torch.zeros(CN_test_image.shape[0])
EMCI_test_label = torch.ones(EMCI_test_image.shape[0])*1
LMCI_test_label = torch.ones(LMCI_test_image.shape[0])*2
AD_test_label = torch.ones(AD_test_image.shape[0])*3

test_label = torch.cat((CN_test_label, EMCI_test_label, LMCI_test_label, AD_test_label))
print(test_label.shape)


# In[21]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)


# In[22]:


import torchvision
print(torchvision.__version__)


# # Hard

# In[24]:


cn = np.load("./weights/hierarchical/cn_external_copy1.npy")
emci = np.load("./weights/hierarchical/emci_external_copy1.npy")
lmci = np.load("./weights/hierarchical/lmci_external_copy1.npy")
ad = np.load("./weights/hierarchical/ad_external_copy1.npy")
print(len(cn))
print(len(emci))
print(len(lmci))
print(len(ad))


# In[25]:


predict_label = [0]*932
for i in range(len(predict_label)):
    for j in emci:
        if i == j:
            predict_label[i] = 1
    for k in lmci:
        if i == k:
            predict_label[i] = 2
    for l in ad:
        if i == l:
            predict_label[i] = 3


# In[26]:


test_acc = 0.0
test_acc += np.sum(predict_label == external_label.numpy())
print(test_acc / 932)


# In[30]:


confmat = confusion_matrix(y_true=external_label.numpy(), y_pred=predict_label)
# Confmat = [confmat[0]/65, confmat[1]/71, confmat[2]/68]

df_cm = pd.DataFrame(confmat, index = [i for i in ["CN", "EMCI", "LMCI", "AD"]],
                     columns = [i for i in ["CN", "EMCI", "LMCI", "AD"]])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g', annot_kws={"size":20})
plt.xlabel('predicted label')        
plt.ylabel('true label')
plt.title('Confusion Matrix')
plt.show()


# In[28]:


#Sensitivity
Class = ["CN", "EMCI", "LMCI", "AD"]
for i in range(4):
  print(Class[i], end=": ")
  print(confmat[i][i]/sum(confmat[i]))


# In[29]:


#Specificity
for i in range(4):
  print(Class[i], end=": ")
  print((np.trace(confmat) - confmat[i][i])/(sum(sum(confmat)) - sum(confmat[i])))


# In[ ]:





# # Soft

# In[61]:


cn_others = np.load("./weights/hierarchical/300epochs/ensemble_hierarchical/cn_others_scores_test.npy")
mci_ad = np.load("./weights/hierarchical/300epochs/ensemble_hierarchical/mci_ad_scores_test.npy")
emci_lmci = np.load("./weights/hierarchical/300epochs/ensemble_hierarchical/emci_lmci_scores_test.npy")
cn_others = torch.from_numpy(cn_others)
mci_ad = torch.from_numpy(mci_ad)
emci_lmci = torch.from_numpy(emci_lmci)
print(cn_others.shape)
print(mci_ad.shape)
print(emci_lmci.shape)


# In[62]:


cn = cn_others[:,0]
emci = cn_others[:,1] * mci_ad[:,0] * emci_lmci[:,0]
lmci = cn_others[:,1] * mci_ad[:,0] * emci_lmci[:,1]
ad = cn_others[:,1] * mci_ad[:,1]


# In[63]:


cn = cn.reshape((932,1))
emci = emci.reshape((932,1))
lmci = lmci.reshape((932,1))
ad = ad.reshape((932,1))


# In[64]:


scores = torch.cat((cn, emci, lmci, ad), 1)


# In[65]:


scores.shape


# In[67]:


test_acc = 0.0
predict_label = np.argmax(scores.cpu().data.numpy(), axis=1)
test_acc += np.sum(predict_label == test_label.numpy())


# In[68]:


test_acc / len(test_image)


# In[69]:


confmat = confusion_matrix(y_true=test_label.numpy(), y_pred=predict_label)
# Confmat = [confmat[0]/65, confmat[1]/71, confmat[2]/68]

df_cm = pd.DataFrame(confmat, index = [i for i in ["CN", "EMCI", "LMCI", "AD"]],
                     columns = [i for i in ["CN", "EMCI", "LMCI", "AD"]])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g', annot_kws={"size":20})
plt.xlabel('predicted label')        
plt.ylabel('true label')
plt.title('Confusion Matrix')
plt.show()


# In[70]:


#Sensitivity
Class = ["CN", "EMCI", "LMCI", "AD"]
for i in range(4):
  print(Class[i], end=": ")
  print(confmat[i][i]/sum(confmat[i]))


# In[71]:


#Specificity
for i in range(4):
  print(Class[i], end=": ")
  print((np.trace(confmat) - confmat[i][i])/(sum(sum(confmat)) - sum(confmat[i])))


# In[ ]:




