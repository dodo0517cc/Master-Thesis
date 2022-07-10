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


# In[17]:


del(AD_image)
del(CN_image)
del(EMCI_image)
del(LMCI_image)


# # AD + CN + MCI

# In[18]:


train_image = torch.cat((CN_train_image, AD_train_image), 0)
print(train_image.shape)

CN_train_label = torch.zeros(CN_train_image.shape[0])
other_train_label = torch.ones(AD_train_image.shape[0])

train_label = torch.cat((CN_train_label, other_train_label))
print(train_label.shape)


# In[19]:


val_image = torch.cat((CN_val_image, AD_val_image), 0)
print(val_image.shape)

CN_val_label = torch.zeros(CN_val_image.shape[0])
other_val_label = torch.ones(AD_val_image.shape[0])

val_label = torch.cat((CN_val_label, other_val_label))
print(val_label.shape)


# In[20]:


test_image = torch.cat((CN_test_image, AD_test_image), 0)
print(test_image.shape)

CN_test_label = torch.zeros(CN_test_image.shape[0])
other_test_label = torch.ones(AD_test_image.shape[0])

test_label = torch.cat((CN_test_label, other_test_label))
print(test_label.shape)


# # Dataloader

# In[21]:


from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler


# In[22]:


# from imblearn.over_sampling import SMOTE
# sm = SMOTE(random_state=42)
# train_image_smote = train_image.reshape(len(train_image),-1)
# print(train_image_smote.shape)
# X_res, y_res = sm.fit_resample(train_image_smote.float(), train_label.long())
# X_res = X_res.reshape(-1,80,110,100)
# print(X_res.shape)


# In[23]:


batch_size = 8
train_loader = DataLoader(TensorDataset(train_image.float(), train_label.long()), batch_size=8, shuffle = True)
val_loader = DataLoader(TensorDataset(val_image.float(), val_label.long()), batch_size=batch_size, shuffle = False)
test_loader = DataLoader(TensorDataset(test_image.float(), test_label.long()), batch_size=batch_size, shuffle = False)


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
make_weights_for_balanced_classes([int(i) for i in train_label],2)
# train_weights = make_weights_for_balanced_classes([int(i) for i in train_label],4)
# train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights,len(train_weights), replacement=True) 


# # Augmentation

# In[25]:


# !pip install torchio
import torchio as tio


# In[26]:


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


# In[27]:


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


# # Train

# In[28]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)


# In[29]:


import torchvision
print(torchvision.__version__)


# In[30]:


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
    def forward(self, x): # text = [batch size,sent_length]
        logits = self.cnn(x)
        return logits


# In[31]:


class FocalLoss(nn.modules.loss._WeightedLoss):
  def __init__(self, weight = None, gamma = 2, reduction = 'mean'):
        super(FocalLoss, self).__init__(weight, reduction = reduction)
        self.gamma = gamma
  def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction = self.reduction, weight = self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss * 2.0).mean()
        return focal_loss


# In[32]:


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)


# In[33]:


# set seed to get the same results
seed = 456
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# In[34]:


if __name__ == '__main__':
    criterion = FocalLoss(torch.FloatTensor([1.467860048820179, 3.137391304347826]).to(device))
#     criterion = nn.CrossEntropyLoss()
    model = CNNRNN_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr = 5e-5, weight_decay=3e-5)
    loss_resnet18_pre = []
    val_loss_resnet18_pre = []

    train_acc_resnet18_pre = []
    val_acc_resnet18_pre = []
    
    num_epochs = 300
    max_acc = 0.0  
    correct_idx = []
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        PATH = './onevsone/300epochs/weight_2class_cn_ad.pkl'
        tp_train = []
#         scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.9)
        model.train()
        for i, (data, label) in enumerate(train_loader):

          optimizer.zero_grad()
          # forward
          data = augmentation(data)
          data = dim1_to_dim3(data)
          outputs = model(data.to(device))
        
          topv, topi = outputs.topk(1) 
          # loss
          loss = criterion(outputs, label.to(device))
          train_loss += loss.item()    
          # accuracy
                
          train_acc += np.sum(np.argmax(outputs.cpu().data.numpy(), axis=1) == label.numpy())
          # update
          loss.backward()
          optimizer.step()
#         scheduler.step()
        model.eval()
        with torch.no_grad():
          val_loss = 0.0
          val_acc = 0.0
          tp_val = []
          for i, (data, label) in enumerate(val_loader):

            # forward
            data = dim1_to_dim3(data)
            outputs = model(data.to(device))
            topv, topi = outputs.topk(1) 
                    
            loss = criterion(outputs, label.to(device))
            val_loss += loss.item()
                
            val_acc += np.sum(np.argmax(outputs.cpu().data.numpy(), axis=1) == label.numpy())

        if val_acc/len(val_image) > max_acc:
            max_acc = val_acc/len(val_image)
            torch.save(model.state_dict(), PATH)
        loss_resnet18_pre.append(train_loss/len(train_image))
        val_loss_resnet18_pre.append(val_loss/len(val_image)) 
        train_acc_resnet18_pre.append(train_acc/len(train_image))
        val_acc_resnet18_pre.append(val_acc/len(val_image))
        
        if epoch == 149:  
            torch.save(model.state_dict(), './onevsone/300epochs/weight_2class_cn_ad_150.pkl')
            
        if (epoch+1) % 1 == 0:
          print('Epoch[{}/{}],Loss:{:.4f},Train Accuracy:{:.2f},Valid Loss:{:.4f},Valid Accuracy:{:.2f}'
          .format(epoch+1,num_epochs,train_loss/len(train_image),train_acc/len(train_image),val_loss/len(val_image),val_acc/len(val_image)))


# In[35]:


history = {"train_loss": loss_resnet18_pre,
          "val_loss": val_loss_resnet18_pre,
          "train_acc": train_acc_resnet18_pre,
          "val_acc": val_acc_resnet18_pre}


# In[36]:


pd.DataFrame(history).to_csv("./onevsone/300epochs/history_cn_ad.csv")


# In[37]:


plt.plot(loss_resnet18_pre, label='train')
plt.plot(val_loss_resnet18_pre, label='val')
plt.legend(loc='upper right')
plt.title("Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()


# In[38]:


plt.plot(train_acc_resnet18_pre, label='train')
plt.plot(val_acc_resnet18_pre, label='val')
plt.legend(loc='upper left')
plt.title("Accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()


# In[39]:


plt.plot(np.array(loss_resnet18_pre)*10, label='train_loss')
plt.plot(train_acc_resnet18_pre, label='train_acc')
plt.plot(val_acc_resnet18_pre, label='val_acc')
plt.legend(loc='upper left')
plt.title("Accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()


# # Test

# In[40]:


# model = CNNRNN_model().to(device)
# PATH = './weight_2class_cn_ad.pkl'
model.load_state_dict(torch.load(PATH))


# In[41]:


pred_label_pre = []
tp_test = []
test_acc = 0.0
model.eval()
for i, (data, label) in enumerate(test_loader):

      # forward
      data = dim1_to_dim3(data)
      outputs = model(data.to(device))
      score = torch.softmax(outputs, dim=1).cpu().data.numpy()
      if i == 0:
        scores = score
      if i > 0:
        scores = np.vstack((scores, score))
        
      topv, topi = outputs.topk(1)   

      test_acc += np.sum(np.argmax(outputs.cpu().data.numpy(), axis=1) == label.numpy())

      pred = np.argmax(outputs.cpu().data.numpy(), axis=1)
      for j in range(len(pred)):
        pred_label_pre.append(pred[j])

      print(np.argmax(outputs.cpu().data.numpy(), axis=1))
      print(label)
      print('')

      del(outputs)
print(test_acc/len(test_image))


# # Confusion Matrix

# In[42]:


confmat = confusion_matrix(y_true=test_label, y_pred=pred_label_pre)
# Confmat = [confmat[0]/65, confmat[1]/71, confmat[2]/68]

df_cm = pd.DataFrame(confmat, index = [i for i in ["CN", "AD"]],
                     columns = [i for i in ["CN", "AD"]])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g', annot_kws={"size":20})
plt.xlabel('predicted label')        
plt.ylabel('true label')
plt.title('Confusion Matrix')
plt.show()


# In[43]:


#Sensitivity
Class = ["CN", "AD"]
for i in range(2):
  print(Class[i], end=": ")
  print(confmat[i][i]/sum(confmat[i]))


# In[44]:


#Specificity
for i in range(2):
  print(Class[i], end=": ")
  print((np.trace(confmat) - confmat[i][i])/(sum(sum(confmat)) - sum(confmat[i])))


# # ROC Curve

# In[45]:


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


# In[46]:


num_classes = 2
binary_label = label_binarize(test_label.cpu().data.numpy(), classes=list(range(num_classes))) # num_classes=10
a = []
for i in test_label.cpu().data.numpy():
    if i == 0:
        a.append(1)
    else:
        a.append(0)
fpr = {}
tpr = {}
roc_auc = {}
fpr[0], tpr[0], _ = roc_curve(binary_label[:,0], scores[:, 1])
fpr[1], tpr[1], _ = roc_curve(a, scores[:, 0])
roc_auc[0] = auc(fpr[0], tpr[0])
roc_auc[1] = auc(fpr[1], tpr[1])

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


# In[47]:


plt.figure(figsize=(8, 8))
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='ROC curve (area = {0:0.3f})'.format(roc_auc["macro"]),
         color='navy', linewidth=4)

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




