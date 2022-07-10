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


# In[16]:


del(AD_image)
del(CN_image)
del(EMCI_image)
del(LMCI_image)


# # Split data

# In[17]:


train_image = torch.cat((CN_train_image, EMCI_train_image, LMCI_train_image, AD_train_image), 0)
print(train_image.shape)

CN_train_label = torch.zeros(CN_train_image.shape[0])
EMCI_train_label = torch.ones(EMCI_train_image.shape[0])
LMCI_train_label = torch.ones(LMCI_train_image.shape[0])*2
AD_train_label = torch.ones(AD_train_image.shape[0])*3
train_label = torch.cat((CN_train_label, EMCI_train_label, LMCI_train_label, AD_train_label))
print(train_label.shape)


# In[18]:


val_image = torch.cat((CN_val_image, EMCI_val_image, LMCI_val_image, AD_val_image), 0)
print(val_image.shape)

CN_val_label = torch.zeros(CN_val_image.shape[0])
EMCI_val_label = torch.ones(EMCI_val_image.shape[0])
LMCI_val_label = torch.ones(LMCI_val_image.shape[0])*2
AD_val_label = torch.ones(AD_val_image.shape[0])*3

val_label = torch.cat((CN_val_label, EMCI_val_label, LMCI_val_label, AD_val_label))
print(val_label.shape)


# In[19]:


test_image = torch.cat((CN_test_image, EMCI_test_image, LMCI_test_image, AD_test_image), 0)
print(test_image.shape)

CN_test_label = torch.zeros(CN_test_image.shape[0])
EMCI_test_label = torch.ones(EMCI_test_image.shape[0])*1
LMCI_test_label = torch.ones(LMCI_test_image.shape[0])*2
AD_test_label = torch.ones(AD_test_image.shape[0])*3

test_label = torch.cat((CN_test_label, EMCI_test_label, LMCI_test_label, AD_test_label))
print(test_label.shape)


# In[20]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)


# In[21]:


import torchvision
print(torchvision.__version__)


# # Dataloader

# In[22]:


from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler


# In[23]:


batch_size = 4
train_loader = DataLoader(TensorDataset(torch.tensor(range(len(train_image))), train_image.float(), train_label.long()), batch_size=batch_size, shuffle = True)
val_loader = DataLoader(TensorDataset(torch.tensor(range(len(val_image))), val_image.float(), val_label.long()), batch_size=batch_size, shuffle = False)
test_loader = DataLoader(TensorDataset(test_image.float(), test_label.long()), batch_size=8, shuffle = False)


# In[24]:


def make_weights_for_balanced_classes(stage, nclasses):                        
    count = [0] * nclasses                                                      
    for item in stage:                                                         
        count[item] += 1                                                    
    weight_per_class = [0.] * nclasses
    N = sum(count)                                                  
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i]) 
    weight = [0] * len(stage)
    for idx, val in enumerate(stage):                                          
        weight[idx] = weight_per_class[val] 
    return weight_per_class, weight
make_weights_for_balanced_classes([int(i) for i in train_label],4)
# train_weights = make_weights_for_balanced_classes([int(i) for i in train_label],4)
# train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights,len(train_weights), replacement=True) 


# In[25]:


# !pip install torchio
import torchio as tio 


# In[26]:


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


# In[27]:


def augmentation(data):
  hi = random.randint(5,20)
  hf = random.randint(15,40)
  di = random.randint(10,30)
  df = random.randint(10,30)
  transform_crop = tio.Compose([
              tio.Crop(cropping=(0,0,hi,hf,di,df)),
              tio.CropOrPad(target_shape=(16, 128, 106), padding_mode=-1)
  ], p=0.7)

  transform_aug = tio.Compose([
#               tio.RandomSwap(patch_size=20, num_iterations=8, p=0.9),
              tio.RandomFlip(p=0.5),
              tio.RandomNoise(p=0.7),
              tio.RandomAffine(scales=(0.9,1.2),
                              degrees=15),
              #tio.RandomElasticDeformation(p=0.5),
              #tio.RandomMotion(p=0.5),
              tio.RandomBiasField(p=0.5)
  ])

  #transform = tio.Compose([transform_crop, transform_aug])
  transform = tio.Compose([transform_aug])
  tran = transform(data)

  return tran


# # Hierarchical

# ## External Test

# In[28]:


def hierarchical_model(cn, emci, lmci, ad):
    length = len(cn) + len(emci) + len(lmci) + len(ad)
    predict_label = [0]*length
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
    return predict_label


# # one versus one (soft)

# In[29]:


class CNNRNN_model1(nn.Module):
    def __init__(self):
        super(CNNRNN_model1, self).__init__()
        self.cnn = models.video.r3d_18(pretrained=True)
        self.features = nn.Sequential(*list(self.cnn.children())[:-1])
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(nn.Linear(num_ftrs, 64),
                                    nn.BatchNorm1d(64),
                                    nn.SELU(),
                                    nn.Dropout(0.3),
                                    nn.Linear(64, 2))
    def forward(self, x): # text = [batch size,sent_length]
 
        logits = self.cnn(x)
        return logits


# In[30]:


class CNNRNN_model2(nn.Module):
    def __init__(self):
        super(CNNRNN_model2, self).__init__()
        self.cnn = models.video.r3d_18(pretrained=True)
        self.features = nn.Sequential(*list(self.cnn.children())[:-1])
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
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(nn.Linear(num_ftrs, 128),
                                    nn.BatchNorm1d(128),
                                    nn.SELU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(128, 2))
    def forward(self, x): # text = [batch size,sent_length]
        logits = self.cnn(x)
        return logits


# In[31]:


class CNNRNN_model3(nn.Module):
    def __init__(self):
        super(CNNRNN_model3, self).__init__()
        self.cnn = models.video.r3d_18(pretrained=True)
        self.features = nn.Sequential(*list(self.cnn.children())[:-1])
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(nn.Linear(num_ftrs, 128),
                                    nn.BatchNorm1d(128),
                                    nn.SELU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(128, 2))
    def forward(self, x): # text = [batch size,sent_length] 
        logits = self.cnn(x)
        return logits


# In[32]:


class CNNRNN_model4(nn.Module):
    def __init__(self):
        super(CNNRNN_model4, self).__init__()
        self.cnn = models.video.r3d_18(pretrained=True)
        self.features = nn.Sequential(*list(self.cnn.children())[:-1])
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(nn.Linear(num_ftrs, 128),
                                    nn.BatchNorm1d(128),
                                    nn.SELU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(128, 2))
    def forward(self, x): # text = [batch size,sent_length] 
        logits = self.cnn(x)
        return logits


# In[33]:


class CNNRNN_model5(nn.Module):
    def __init__(self):
        super(CNNRNN_model5, self).__init__()
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


# In[34]:


class CNNRNN_model6(nn.Module):
    def __init__(self):
        super(CNNRNN_model6, self).__init__()
        self.cnn = models.video.r3d_18(pretrained=True)
        self.features = nn.Sequential(*list(self.cnn.children())[:-1])
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(nn.Linear(num_ftrs, 64),
                                    nn.BatchNorm1d(64),
                                    nn.SELU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(64, 2))
    def forward(self, x): # text = [batch size,sent_length]  
        logits = self.cnn(x)
        return logits


# In[35]:


ovo_model1 = CNNRNN_model1().to(device)
ovo_model1.load_state_dict(torch.load('./onevsone/300epochs/weight_2class_lmci_ad.pkl'))
ovo_model2 = CNNRNN_model2().to(device)
ovo_model2.load_state_dict(torch.load('./onevsone/300epochs/weight_2class_emci_ad.pkl'))
ovo_model3 = CNNRNN_model3().to(device)
ovo_model3.load_state_dict(torch.load('./onevsone/300epochs/weight_2class_cn_ad.pkl'))
ovo_model4 = CNNRNN_model4().to(device)
ovo_model4.load_state_dict(torch.load('./onevsone/300epochs/weight_2class_cn_emci.pkl'))
ovo_model5 = CNNRNN_model5().to(device)
ovo_model5.load_state_dict(torch.load('./onevsone/300epochs/weight_2class_cn_lmci.pkl'))
ovo_model6 = CNNRNN_model6().to(device)
ovo_model6.load_state_dict(torch.load('./onevsone/300epochs/weight_2class_emci_lmci.pkl'))


# In[36]:


ovo_model1.eval()
ovo_model2.eval()
ovo_model3.eval()
ovo_model4.eval()
ovo_model5.eval()
ovo_model6.eval()


# In[37]:


def onevsone_model(data):
      scores = np.zeros(data.shape[0]*4).reshape((data.shape[0],4))
      # model1
      outputs = ovo_model1(data.to(device))
      score = torch.softmax(outputs, dim=1).cpu().data.numpy()  
      scores[:,2] += score[:,0]
      scores[:,3] += score[:,1] 
      # model2
      outputs = ovo_model2(data.to(device))
      score = torch.softmax(outputs, dim=1).cpu().data.numpy()  
      scores[:,1] += score[:,0]
      scores[:,3] += score[:,1] 
      # model3
      outputs = ovo_model3(data.to(device))
      score = torch.softmax(outputs, dim=1).cpu().data.numpy()  
      scores[:,0] += score[:,0]
      scores[:,3] += score[:,1]         
      # model4
      outputs = ovo_model4(data.to(device))
      score = torch.softmax(outputs, dim=1).cpu().data.numpy()  
      scores[:,0] += score[:,0]
      scores[:,1] += score[:,1]         
      # model5
      outputs = ovo_model5(data.to(device))
      score = torch.softmax(outputs, dim=1).cpu().data.numpy()  
      scores[:,0] += score[:,0]
      scores[:,2] += score[:,1]         
      # model6
      outputs = ovo_model6(data.to(device))
      score = torch.softmax(outputs, dim=1).cpu().data.numpy()  
      scores[:,1] += score[:,0]
      scores[:,2] += score[:,1] 
        
#       scores =  torch.tensor([np.array(x)/3 for x in scores]).to(device)   
      return scores  


# # one vesus all

# In[38]:


class CNNRNN_model(nn.Module):
    def __init__(self):
        super(CNNRNN_model, self).__init__()
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


# In[39]:


ova_model1 = CNNRNN_model().to(device)
ova_model1.load_state_dict(torch.load('./onevsall/300epochs/weight_cn_others.pkl'))
ova_model2 = CNNRNN_model().to(device)
ova_model2.load_state_dict(torch.load('./onevsall/300epochs/weight_emci_others.pkl'))
ova_model3 = CNNRNN_model().to(device)
ova_model3.load_state_dict(torch.load('./onevsall/300epochs/weight_lmci_others.pkl'))
ova_model4 = CNNRNN_model().to(device)
ova_model4.load_state_dict(torch.load('./onevsall/300epochs/weight_ad_others.pkl'))


# In[40]:


ova_model1.eval()
ova_model2.eval()
ova_model3.eval()
ova_model4.eval()


# In[41]:


def onevsall_model(data):
      scores = np.zeros(data.shape[0]*4).reshape((data.shape[0],4))
      # model1
      outputs = ova_model1(data.to(device))
      score = torch.softmax(outputs, dim=1).cpu().data.numpy()  
      scores[:,0] += score[:,0]  
      # model2
      outputs = ova_model2(data.to(device))
      score = torch.softmax(outputs, dim=1).cpu().data.numpy()  
      scores[:,1] += score[:,0]
      # model3
      outputs = ova_model3(data.to(device))
      score = torch.softmax(outputs, dim=1).cpu().data.numpy()  
      scores[:,2] += score[:,0]      
      # model4
      outputs = ova_model4(data.to(device))
      score = torch.softmax(outputs, dim=1).cpu().data.numpy()  
      scores[:,3] += score[:,0]
      return scores


# # 4class

# In[42]:


class CNNRNN_model(nn.Module):
    def __init__(self):
        super(CNNRNN_model, self).__init__()
        self.cnn = models.video.r3d_18(pretrained=True)
        self.features = nn.Sequential(*list(self.cnn.children())[:-1])
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(nn.Linear(num_ftrs, 128),
                                    nn.BatchNorm1d(128),
                                    nn.SELU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(128, 4))
    def forward(self, x): # text = [batch size,sent_length]  
        logits = self.cnn(x)
        return logits


# In[43]:


flat_model = CNNRNN_model().to(device)
PATH = './weights/300epochs/weight_4class_noswap_fc_affine_learningcurves_300.pkl'
flat_model.load_state_dict(torch.load(PATH))
flat_model.eval()


# ## Hierarchical

# In[62]:


class CNNRNN_model1(nn.Module):
    def __init__(self):
        super(CNNRNN_model1, self).__init__()
        self.cnn = models.video.r3d_18(pretrained=True)
        self.features = nn.Sequential(*list(self.cnn.children())[:-1])
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(nn.Linear(num_ftrs, 128),
                                    nn.BatchNorm1d(128),
                                    nn.SELU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(128, 2))
    def forward(self, x): # text = [batch size,sent_length]  
        logits = self.cnn(x)
        return logits


# In[63]:


class CNNRNN_model2(nn.Module):
    def __init__(self):
        super(CNNRNN_model2, self).__init__()
        self.cnn = models.video.r3d_18(pretrained=True)
        self.features = nn.Sequential(*list(self.cnn.children())[:-1])
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(nn.Linear(num_ftrs, 64),
                                    nn.BatchNorm1d(64),
                                    nn.SELU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(64, 2))
    def forward(self, x): # text = [batch size,sent_length]
        logits = self.cnn(x)
        return logits


# In[64]:


class CNNRNN_model3(nn.Module):
    def __init__(self):
        super(CNNRNN_model3, self).__init__()
        self.cnn = models.video.r3d_18(pretrained=True)
        self.features = nn.Sequential(*list(self.cnn.children())[:-1])
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(nn.Linear(num_ftrs, 64),
                                    nn.BatchNorm1d(64),
                                    nn.SELU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(64, 2))
    def forward(self, x): # text = [batch size,sent_length]  
        logits = self.cnn(x)
        return logits


# In[65]:


hirearchical_model1 = CNNRNN_model1().to(device)
hirearchical_model2 = CNNRNN_model2().to(device)
hirearchical_model3 = CNNRNN_model3().to(device)
PATH1 = './weights/hierarchical/300epochs/weight_cn_others_learningcurves.pkl'
PATH2 = './weights/hierarchical/300epochs/weight_mci_ad_learningcurves.pkl'
PATH3 = './weights/hierarchical/300epochs/weight_emci_lmci_learningcurves.pkl'
hirearchical_model1.load_state_dict(torch.load(PATH1))
hirearchical_model2.load_state_dict(torch.load(PATH2))
hirearchical_model3.load_state_dict(torch.load(PATH3))
hirearchical_model1.eval()
hirearchical_model2.eval()
hirearchical_model3.eval()


# # Stacking soft

# ## hierarchical

# In[44]:


cn_others = np.load("./weights/hierarchical/300epochs/ensemble_hierarchical/cn_others_scores.npy")
cn_others = torch.from_numpy(cn_others)
mci_ad = np.load("./weights/hierarchical/300epochs/ensemble_hierarchical/mci_ad_scores.npy")
mci_ad = torch.from_numpy(mci_ad)
emci_lmci = np.load("./weights/hierarchical/300epochs/ensemble_hierarchical/emci_lmci_scores.npy")
emci_lmci = torch.from_numpy(emci_lmci)
for i in range(300):
    cn = torch.unsqueeze(cn_others[i][:,0],1)
    emci = torch.unsqueeze(cn_others[i][:,1] * mci_ad[i][:,0] * emci_lmci[i][:,0],1)
    lmci = torch.unsqueeze(cn_others[i][:,1] * mci_ad[i][:,0] * emci_lmci[i][:,1],1)
    ad = torch.unsqueeze(cn_others[i][:,1] * mci_ad[i][:,1],1)
    scores = torch.cat((cn, emci, lmci, ad), 1)
    if i == 0:
        hierarchical_train = scores
    if i > 0:
        hierarchical_train = np.vstack((hierarchical_train, scores))


# In[45]:


hierarchical_train = hierarchical_train.reshape((300,2981,4))
print(hierarchical_train.shape)


# In[46]:


cn_others = np.load("./weights/hierarchical/300epochs/ensemble_hierarchical/cn_others_scores_val.npy")
cn_others = torch.from_numpy(cn_others)
mci_ad = np.load("./weights/hierarchical/300epochs/ensemble_hierarchical/mci_ad_scores_val.npy")
mci_ad = torch.from_numpy(mci_ad)
emci_lmci = np.load("./weights/hierarchical/300epochs/ensemble_hierarchical/emci_lmci_scores_val.npy")
emci_lmci = torch.from_numpy(emci_lmci)


# In[47]:


cn = torch.unsqueeze(cn_others[:,0],1)
emci = torch.unsqueeze(cn_others[:,1] * mci_ad[:,0] * emci_lmci[:,0],1)
lmci = torch.unsqueeze(cn_others[:,1] * mci_ad[:,0] * emci_lmci[:,1],1)
ad = torch.unsqueeze(cn_others[:,1] * mci_ad[:,1],1)


# In[48]:


hierarchical_val = torch.cat((cn, emci, lmci, ad), 1)
print(hierarchical_val.shape)


# In[49]:


hierarchical_train = torch.tensor(hierarchical_train)
hierarchical_train.shape


# In[50]:


cn_others = np.load("./weights/hierarchical/300epochs/ensemble_hierarchical/cn_others_scores_test.npy")
cn_others = torch.from_numpy(cn_others)
mci_ad = np.load("./weights/hierarchical/300epochs/ensemble_hierarchical/mci_ad_scores_test.npy")
mci_ad = torch.from_numpy(mci_ad)
emci_lmci = np.load("./weights/hierarchical/300epochs/ensemble_hierarchical/emci_lmci_scores_test.npy")
emci_lmci = torch.from_numpy(emci_lmci)
cn = torch.unsqueeze(cn_others[:,0],1)
emci = torch.unsqueeze(cn_others[:,1] * mci_ad[:,0] * emci_lmci[:,0],1)
lmci = torch.unsqueeze(cn_others[:,1] * mci_ad[:,0] * emci_lmci[:,1],1)
ad = torch.unsqueeze(cn_others[:,1] * mci_ad[:,1],1)
hierarchical_test = torch.cat((cn, emci, lmci, ad), 1)
print(hierarchical_test.shape)


# ## OVO

# In[51]:


ovo_train = np.load("./ensemble/onevsone_train_scores.npy")
ovo_train = torch.from_numpy(ovo_train)
print(ovo_train.shape)
ovo_val = np.load("./ensemble/onevsone_val_scores.npy")
ovo_val = torch.from_numpy(ovo_val)
print(ovo_val.shape)
ovo_test = np.load("./ensemble/onevsone_test_scores.npy")
ovo_test = torch.from_numpy(ovo_test)
print(ovo_test.shape)


# ## OVA

# In[52]:


ova_train = np.load("./ensemble/onevsall_train_scores.npy")
ova_train = torch.from_numpy(ova_train)
print(ova_train.shape)
ova_val = np.load("./ensemble/onevsall_val_scores.npy")
ova_val = torch.from_numpy(ova_val)
print(ova_val.shape)
ova_test = np.load("./ensemble/onevsall_test_scores.npy")
ova_test = torch.from_numpy(ova_test)
print(ova_test.shape)


# ## FLAT

# In[53]:


flat_train = np.load("./ensemble/flat_train_scores.npy")
flat_train = torch.from_numpy(flat_train)
print(flat_train.shape)
flat_val = np.load("./ensemble/flat_val_scores.npy")
flat_val = torch.from_numpy(flat_val)
print(flat_val.shape)
flat_test = np.load("./ensemble/flat_test_scores.npy")
flat_test = torch.from_numpy(flat_test)
print(flat_test.shape)


# In[54]:


save_label = np.load("./ensemble/train_label.npy")
save_label = torch.from_numpy(save_label)


# In[55]:


class Stacking_model(torch.nn.Module):
  def __init__(self):
    super(Stacking_model, self).__init__()

    self.layer_1 = nn.Sequential(
        nn.Linear(16, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.5)
        )

    self.layer_2 = torch.nn.Linear(64, 16)
    
    self.layer_3 = nn.Sequential(
        nn.Linear(64,16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(16,4)
        )

  def forward(self, hierarchical, onevsall, onevsone, flat):
    
    x = torch.cat((hierarchical, onevsall, onevsone, flat), 1)
    ris = self.layer_1(x)
    ris = self.layer_3(ris)

    return ris


# In[56]:


batch_size = 16
# train_loader = DataLoader(TensorDataset(ovo_train.float(), ova_train.float(), flat_train.float(), a.long()), batch_size=batch_size, shuffle = True)
val_loader = DataLoader(TensorDataset(hierarchical_val.float(), ovo_val[0].float(), ova_val.float(), flat_val[0].float(), val_label.long()), batch_size=batch_size, shuffle = False)


# In[59]:


max_acc = 0.0
num_epochs = 150
stacking_model = Stacking_model().to(device)
optimizer = optim.Adam(stacking_model.parameters(), lr = 1e-4)
# criterion = FocalLoss(weight=torch.FloatTensor([2.4255492270138324, 3.851421188630491, 7.397022332506204, 5.184347826086957]).to(device))
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([2.4255492270138324, 3.851421188630491, 7.397022332506204, 5.184347826086957]).to(device))
loss_resnet18_pre = []
val_loss_resnet18_pre = []

train_acc_resnet18_pre = []
val_acc_resnet18_pre = []
for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0.0
    stacking_model.train()
    train_loader = DataLoader(TensorDataset(hierarchical_train[epoch].float(), ovo_train[epoch].float(), ova_train[epoch].float(), flat_train[epoch].float(), save_label[epoch].long()), batch_size=batch_size, shuffle = False)
    for i, (hierarchical, onevsone, onevsall, flat, label) in enumerate(train_loader):
#     for i, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()
        # forward
        outputs = stacking_model(hierarchical.to(device), onevsall.to(device), onevsone.to(device), flat.to(device))
        # loss
        loss = criterion(outputs, label.to(device))
        train_loss += loss.item()
        # accuracy
        train_acc += np.sum(np.argmax(outputs.cpu().data.numpy(), axis=1) == label.numpy())

        # update
        loss.backward()
        optimizer.step()
#         scheduler.step()
    stacking_model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_acc = 0.0
        for i, (hierarchical, onevsone, onevsall, flat, label) in enumerate(val_loader):

          # forward
            outputs = stacking_model(hierarchical.to(device), onevsall.to(device), onevsone.to(device), flat.to(device))
            loss = criterion(outputs, label.to(device))
            val_loss += loss.item()

            val_acc += np.sum(np.argmax(outputs.cpu().data.numpy(), axis=1) == label.numpy())
    if val_acc/len(val_image) > max_acc:
        max_acc = val_acc/len(val_image)
        torch.save(stacking_model.state_dict(), './stacking_soft_try.pkl')
    loss_resnet18_pre.append(train_loss/len(train_image)) 
    val_loss_resnet18_pre.append(val_loss/len(val_image)) 
    train_acc_resnet18_pre.append(train_acc/len(train_image))
    val_acc_resnet18_pre.append(val_acc/len(val_image))


    if (epoch+1) % 1 == 0:
        print('Epoch[{}/{}],Loss:{:.4f},Train Accuracy:{:.2f},Val Loss:{:.4f},Val Accuracy:{:.2f}'
        .format(epoch+1,num_epochs,train_loss/len(train_image),train_acc/len(train_image), val_loss/len(val_image), val_acc/len(val_image)))


# In[60]:


plt.plot(train_acc_resnet18_pre, label='train_acc')
plt.plot(val_acc_resnet18_pre, label='val_acc')
plt.plot(np.array(loss_resnet18_pre)*10, label='train_loss')
plt.legend(loc='center right')
plt.title("Loss and accuracy curves")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()


# In[62]:


test_loader = DataLoader(TensorDataset(hierarchical_test.float(), ovo_test.float(), ova_test.float(), flat_test.float(), test_label.long()), batch_size=batch_size, shuffle = False)


# In[65]:


PATH = './ensemble/stacking_soft.pkl'
stacking_model.load_state_dict(torch.load(PATH))
pred_label_pre = []
test_acc = 0.0
stacking_model.eval()
for i, (hierarchical, onevsone, onevsall, flat, label) in enumerate(test_loader):

    outputs = stacking_model(hierarchical.to(device), onevsall.to(device), onevsone.to(device), flat.to(device))
    score = torch.softmax(outputs, dim=1).cpu().data.numpy()
    if i == 0:
        scores = score
    if i > 0:
        scores = np.vstack((scores, score))
    test_acc += np.sum(np.argmax(outputs.cpu().data.numpy(), axis=1) == label.numpy())

    pred = np.argmax(outputs.cpu().data.numpy(), axis=1)
    for j in range(len(pred)):
        pred_label_pre.append(pred[j])

    print(np.argmax(outputs.cpu().data.numpy(), axis=1))
    print(label)
    print('')

    del(outputs)
print(test_acc/len(test_image))


# In[66]:


confmat = confusion_matrix(y_true=test_label, y_pred=pred_label_pre)

df_cm = pd.DataFrame(confmat, index = [i for i in ["CN", "EMCI", "LMCI", "AD"]],
                     columns = [i for i in ["CN", "EMCI", "LMCI", "AD"]])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g', annot_kws={"size":20})
plt.xlabel('predicted label')        
plt.ylabel('true label')
plt.title('Confusion Matrix')
plt.show()


# In[67]:


#Sensitivity
Class = ["CN", "EMCI", "LMCI", "AD"]
for i in range(4):
  print(Class[i], end=": ")
  print(confmat[i][i]/sum(confmat[i]))


# In[68]:


#Specificity
for i in range(4):
  print(Class[i], end=": ")
  print((np.trace(confmat) - confmat[i][i])/(sum(sum(confmat)) - sum(confmat[i])))


# In[69]:


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


# In[70]:


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


# In[71]:


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
plt.savefig('Multi-class ROC.jpg', bbox_inches='tight')
plt.show()


# In[ ]:




