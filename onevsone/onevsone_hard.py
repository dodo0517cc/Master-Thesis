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
CN_train_image = CN_image[:idx]
print(CN_train_image.shape)

CN_val_image = CN_image[idx:idx2]
print(CN_val_image.shape)

CN_test_image = CN_image[idx2:]
print(CN_test_image.shape)


# In[13]:


idx = round(EMCI_image.shape[0]*0.8*0.8)
idx2 = round(EMCI_image.shape[0]*0.8)
EMCI_train_image = EMCI_image[:idx]
print(EMCI_train_image.shape)

EMCI_val_image = EMCI_image[idx:idx2]
print(EMCI_val_image.shape)

EMCI_test_image = EMCI_image[idx2:]
print(EMCI_test_image.shape)


# In[14]:


idx = round(LMCI_image.shape[0]*0.8*0.8)
idx2 = round(LMCI_image.shape[0]*0.8)
LMCI_train_image = LMCI_image[:idx]
print(LMCI_train_image.shape)

LMCI_val_image = LMCI_image[idx:idx2]
print(LMCI_val_image.shape)

LMCI_test_image = LMCI_image[idx2:]
print(LMCI_test_image.shape)


# In[15]:


idx = round(AD_image.shape[0]*0.8*0.8)
idx2 = round(AD_image.shape[0]*0.8)
AD_train_image = AD_image[:idx]
print(AD_train_image.shape)

AD_val_image = AD_image[idx:idx2]
print(AD_val_image.shape)

AD_test_image = AD_image[idx2:]
print(AD_test_image.shape)


# In[16]:


del(AD_image)
del(CN_image)
del(EMCI_image)
del(LMCI_image)


# # AD + CN + EMCI + LMCI

# In[17]:


test_image = torch.cat((CN_test_image, EMCI_test_image, LMCI_test_image, AD_test_image), 0)
print(test_image.shape)

CN_test_label = torch.zeros(CN_test_image.shape[0])
EMCI_test_label = torch.ones(EMCI_test_image.shape[0])*1
LMCI_test_label = torch.ones(LMCI_test_image.shape[0])*2
AD_test_label = torch.ones(AD_test_image.shape[0])*3

test_label = torch.cat((CN_test_label, EMCI_test_label, LMCI_test_label, AD_test_label))
print(test_label.shape)


# # Dataloader

# In[18]:


from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler


# In[19]:


test_loader = DataLoader(TensorDataset(test_image.float() ,test_label.long()), batch_size=1, shuffle = False)


# # Augmentation

# In[20]:


# !pip install torchio
import torchio as tio 


# In[21]:


def dim1_to_dim3(image):
  # image.shape = (batch_size, slice, h, w)
  image = image.unsqueeze(1)
  img_shape = np.array(image.shape)
  img_shape[1] = 3
  out_image = torch.zeros(tuple(img_shape))
  out_image[:,0,:,:,:] = image[:,0,:,:,:]
  out_image[:,1,:,:,:] = image[:,0,:,:,:]
  out_image[:,2,:,:,:] = image[:,0,:,:,:]
  
  return out_image


# In[22]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)


# In[23]:


import torchvision
print(torchvision.__version__)


# # LCMI / AD

# In[24]:


class CNN_model1(nn.Module):
    def __init__(self):
        super(CNN_model1, self).__init__()
        self.cnn = models.video.r3d_18(pretrained=True)
        self.features = nn.Sequential(*list(self.cnn.children())[:-1])
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(nn.Linear(num_ftrs, 64),
                                    nn.BatchNorm1d(64),
                                    nn.SELU(),
                                    nn.Dropout(0.3),
                                    nn.Linear(64, 2))
    def forward(self, x):
        logits = self.cnn(x)
        return logits


# # EMCI / AD

# In[25]:


class CNN_model2(nn.Module):
    def __init__(self):
        super(CNN_model2, self).__init__()
        self.cnn = models.video.r3d_18(pretrained=True)
        self.features = nn.Sequential(*list(self.cnn.children())[:-1])
        num_ftrs = self.cnn.fc.in_features
        self.lstm = nn.LSTM(512, 512, 
                           num_layers=2,
                           bidirectional=True,
                           dropout=0,
                           batch_first=True)
        self.fc = nn.Sequential(
                            nn.Linear(1024,512),
                            nn.SELU(),
                            nn.Linear(512,4),
                        )
        self.cnn.fc = nn.Sequential(nn.Linear(num_ftrs, 128),
                                    nn.BatchNorm1d(128),
                                    nn.SELU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(128, 2))
    def forward(self, x): 
        logits = self.cnn(x)
        return logits


# # CN / AD

# In[26]:


class CNN_model3(nn.Module):
    def __init__(self):
        super(CNN_model3, self).__init__()
        self.cnn = models.video.r3d_18(pretrained=True)
        self.features = nn.Sequential(*list(self.cnn.children())[:-1])
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(nn.Linear(num_ftrs, 128),
                                    nn.BatchNorm1d(128),
                                    nn.SELU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(128, 2))
    def forward(self, x):
        logits = self.cnn(x)
        return logits


# # CN / EMCI

# In[27]:


class CNN_model4(nn.Module):
    def __init__(self):
        super(CNN_model4, self).__init__()
        self.cnn = models.video.r3d_18(pretrained=True)
        self.features = nn.Sequential(*list(self.cnn.children())[:-1])
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(nn.Linear(num_ftrs, 128),
                                    nn.BatchNorm1d(128),
                                    nn.SELU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(128, 2))
    def forward(self, x): 
        logits = self.cnn(x)
        return logits


# # CN / LMCI

# In[28]:


class CNN_model5(nn.Module):
    def __init__(self):
        super(CNN_model5, self).__init__()
        self.cnn = models.video.r3d_18(pretrained=True)
        self.features = nn.Sequential(*list(self.cnn.children())[:-1])
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(nn.Linear(num_ftrs, 128),
                                    nn.BatchNorm1d(128),
                                    nn.SELU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(128, 2))
    def forward(self, x): 
        logits = self.cnn(x)
        return logits


# # EMCI / LMCI

# In[29]:


class CNN_model6(nn.Module):
    def __init__(self):
        super(CNN_model6, self).__init__()
        self.cnn = models.video.r3d_18(pretrained=True)
        self.features = nn.Sequential(*list(self.cnn.children())[:-1])
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(nn.Linear(num_ftrs, 64),
                                    nn.BatchNorm1d(64),
                                    nn.SELU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(64, 2))
    def forward(self, x):  
        logits = self.cnn(x)
        return logits


# # Test

# In[30]:


model_lmci_ad = CNN_model1().to(device)
model_lmci_ad.load_state_dict(torch.load('./onevsone/300epochs/weight_2class_lmci_ad.pkl'))

model_emci_ad = CNN_model2().to(device)
model_emci_ad.load_state_dict(torch.load('./onevsone/300epochs/weight_2class_emci_ad.pkl'))

model_cn_ad = CNN_model3().to(device)
model_cn_ad.load_state_dict(torch.load('./onevsone/300epochs/weight_2class_cn_ad.pkl'))

model_cn_emci = CNN_model4().to(device)
model_cn_emci.load_state_dict(torch.load('./onevsone/300epochs/weight_2class_cn_emci.pkl'))

model_cn_lmci = CNN_model5().to(device)
model_cn_lmci.load_state_dict(torch.load('./onevsone/300epochs/weight_2class_cn_lmci.pkl'))

model_emci_lmci = CNN_model6().to(device)
model_emci_lmci.load_state_dict(torch.load('./onevsone/300epochs/weight_2class_emci_lmci.pkl'))


# In[31]:


pred_label_pre = []
test_acc = 0.0
model_lmci_ad.eval()
model_emci_ad.eval()
model_cn_ad.eval()
model_cn_emci.eval()
model_cn_lmci.eval()
model_emci_lmci.eval()
for i, (data, label) in enumerate(test_loader):
      # forward
      data = dim1_to_dim3(data)
      count_label = [0]*4
        
      # model1
      outputs = model_lmci_ad(data.to(device))
      topv, topi = outputs.topk(1)
      pred = np.argmax(outputs.cpu().data.numpy(), axis=1)
      if pred[0] == 0:
        count_label[2] += 1
      else:
        count_label[3] += 1
        
      # model2
      outputs = model_emci_ad(data.to(device))
      topv, topi = outputs.topk(1)
      pred = np.argmax(outputs.cpu().data.numpy(), axis=1)
      if pred[0] == 0:
        count_label[1] += 1
      else:
        count_label[3] += 1
      # model3  
      outputs = model_cn_ad(data.to(device))
      topv, topi = outputs.topk(1)
      pred = np.argmax(outputs.cpu().data.numpy(), axis=1)
      if pred[0] == 0:
        count_label[0] += 1
      else:
        count_label[3] += 1
      # model4  
      outputs =  model_cn_emci(data.to(device))
      topv, topi = outputs.topk(1)
      pred = np.argmax(outputs.cpu().data.numpy(), axis=1)
      if pred[0] == 0:
        count_label[0] += 1
      else:
        count_label[1] += 1    
      # model5
      outputs = model_cn_lmci(data.to(device))
      topv, topi = outputs.topk(1)
      pred = np.argmax(outputs.cpu().data.numpy(), axis=1)
      if pred[0] == 0:
        count_label[0] += 1
      else:
        count_label[2] += 1    
      #model6
      outputs = model_emci_lmci(data.to(device))
      topv, topi = outputs.topk(1)
      pred = np.argmax(outputs.cpu().data.numpy(), axis=1)
      if pred[0] == 0:
        count_label[1] += 1
      else:
        count_label[2] += 1
        
      if i == 0:
        scores = count_label
      if i > 0:
        scores = np.vstack((scores, count_label))
        
      # predict
      predict = np.argmax(count_label)
      test_acc += np.sum(predict == label.numpy())
      pred_label_pre.append(predict)



print(test_acc/len(test_image))


# In[32]:


scores.shape


# # Confusion Matrix

# In[33]:


confmat = confusion_matrix(y_true=test_label, y_pred=pred_label_pre)

df_cm = pd.DataFrame(confmat, index = [i for i in ["CN", "EMCI", "LMCI", "AD"]],
                     columns = [i for i in ["CN", "EMCI", "LMCI", "AD"]])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g', annot_kws={"size":20})
plt.xlabel('predicted label')        
plt.ylabel('true label')
plt.title('Confusion Matrix')
plt.show()


# In[38]:


#Sensitivity
Class = ["CN", "EMCI", "LMCI", "AD"]
for i in range(4):
  print(Class[i], end=": ")
  print(confmat[i][i]/sum(confmat[i]))


# In[39]:


#Specificity
for i in range(4):
  print(Class[i], end=": ")
  print((np.trace(confmat) - confmat[i][i])/(sum(sum(confmat)) - sum(confmat[i])))


# # ROC curve

# In[40]:


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


# In[41]:


num_classes = 4
binary_label = label_binarize(test_label.cpu().data.numpy(), classes=list(range(num_classes))) # num_classes=10

fpr = {}
tpr = {}
roc_auc = {}

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(binary_label[:, i], scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(binary_label.ravel(), scores.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= num_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


# In[42]:


plt.figure(figsize=(8, 8))
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.3f})'.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.3f})'.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

Class = ["CN", "EMCI", "LMCI", "AD"]
for i in range(4):
    plt.plot(fpr[i], tpr[i], lw=2,
             label='ROC curve of {0} (area = {1:0.3f})'.format(Class[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC')
plt.legend(loc="lower right")
# plt.savefig('Multi-class ROC.jpg', bbox_inches='tight')
plt.show()


# In[ ]:




