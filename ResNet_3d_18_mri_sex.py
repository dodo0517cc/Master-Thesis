#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
import os
#import cv2
import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset, SubsetRandomSampler
from torch.utils import model_zoo
import nibabel as nib
# import skimage.io as io
# from skimage.transform import resize
import sys
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.model_selection import KFold


# In[2]:


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# In[3]:


os.chdir('/home/u9285752')


# # Load Data

# In[4]:


CN_image= np.load("./CN_seg_voxel.npy")
CN_image = CN_image[:,10:90,5:115,10:110]
CN_image = torch.from_numpy(CN_image)
CN_image.shape


# In[5]:


EMCI_image= np.load("./EMCI_seg_voxel.npy")
EMCI_image = EMCI_image[:,10:90,5:115,10:110]
EMCI_image = torch.from_numpy(EMCI_image)
EMCI_image.shape


# In[6]:


LMCI_image= np.load("./LMCI_seg_voxel.npy")
LMCI_image = LMCI_image[:,10:90,5:115,10:110]
LMCI_image = torch.from_numpy(LMCI_image)
LMCI_image.shape


# In[7]:


AD_image= np.load("./AD_seg_voxel.npy")
AD_image = AD_image[:,10:90,5:115,10:110]
AD_image = torch.from_numpy(AD_image)
AD_image.shape


# # Split Train & Test 

# In[8]:


idx = round(CN_image.shape[0]*0.8*0.8)
idx2 = round(CN_image.shape[0]*0.8)
CN_train_image = CN_image[:idx]
print(CN_train_image.shape)

CN_val_image = CN_image[idx:idx2]
print(CN_val_image.shape)

CN_test_image = CN_image[idx2:]
print(CN_test_image.shape)


# In[9]:


idx = round(EMCI_image.shape[0]*0.8*0.8)
idx2 = round(EMCI_image.shape[0]*0.8)
EMCI_train_image = EMCI_image[:idx]
print(EMCI_train_image.shape)

EMCI_val_image = EMCI_image[idx:idx2]
print(EMCI_val_image.shape)

EMCI_test_image = EMCI_image[idx2:]
print(EMCI_test_image.shape)


# In[10]:


idx = round(LMCI_image.shape[0]*0.8*0.8)
idx2 = round(LMCI_image.shape[0]*0.8)
LMCI_train_image = LMCI_image[:idx]
print(LMCI_train_image.shape)

LMCI_val_image = LMCI_image[idx:idx2]
print(LMCI_val_image.shape)

LMCI_test_image = LMCI_image[idx2:]
print(LMCI_test_image.shape)


# In[11]:


idx = round(AD_image.shape[0]*0.8*0.8)
idx2 = round(AD_image.shape[0]*0.8)
AD_train_image = AD_image[:idx]
print(AD_train_image.shape)

AD_val_image = AD_image[idx:idx2]
print(AD_val_image.shape)

AD_test_image = AD_image[idx2:]
print(AD_test_image.shape)


# In[12]:


del(AD_image)
del(CN_image)
del(EMCI_image)
del(LMCI_image)


# # AD + CN + MCI

# In[13]:


train_image = torch.cat((CN_train_image, EMCI_train_image, LMCI_train_image, AD_train_image), 0)
print(train_image.shape)

CN_train_label = torch.zeros(CN_train_image.shape[0])
EMCI_train_label = torch.ones(EMCI_train_image.shape[0])
LMCI_train_label = torch.ones(LMCI_train_image.shape[0])*2
AD_train_label = torch.ones(AD_train_image.shape[0])*3
train_label = torch.cat((CN_train_label, EMCI_train_label, LMCI_train_label, AD_train_label))
print(train_label.shape)


# In[14]:


val_image = torch.cat((CN_val_image, EMCI_val_image, LMCI_val_image, AD_val_image), 0)
print(val_image.shape)

CN_val_label = torch.zeros(CN_val_image.shape[0])
EMCI_val_label = torch.ones(EMCI_val_image.shape[0])
LMCI_val_label = torch.ones(LMCI_val_image.shape[0])*2
AD_val_label = torch.ones(AD_val_image.shape[0])*3
val_label = torch.cat((CN_val_label, EMCI_val_label, LMCI_val_label, AD_val_label))
print(val_label.shape)


# In[15]:


test_image = torch.cat((CN_test_image, EMCI_test_image, LMCI_test_image, AD_test_image), 0)
print(test_image.shape)

CN_test_label = torch.zeros(CN_test_image.shape[0])
EMCI_test_label = torch.ones(EMCI_test_image.shape[0])
LMCI_test_label = torch.ones(LMCI_test_image.shape[0])*2
AD_test_label = torch.ones(AD_test_image.shape[0])*3
test_label = torch.cat((CN_test_label, EMCI_test_label, LMCI_test_label, AD_test_label))
print(test_label.shape)


# # df

# In[16]:


CN_df = pd.read_csv('./mmse/CN_APOE.csv')
idx = round(len(CN_df)*0.8*0.8)
idx2 = round(len(CN_df)*0.8)
CN_train_df = CN_df[:idx]
print(len(CN_train_df))

CN_val_df= CN_df[idx:idx2]
print(len(CN_val_df))

CN_test_df = CN_df[idx2:]
print(len(CN_test_df))


# In[17]:


EMCI_df = pd.read_csv('./mmse/EMCI_APOE.csv')
idx = round(len(EMCI_df)*0.8*0.8)
idx2 = round(len(EMCI_df)*0.8)
EMCI_train_df = EMCI_df[:idx]
print(len(EMCI_train_df))

EMCI_val_df = EMCI_df[idx:idx2]
print(len(EMCI_val_df))

EMCI_test_df = EMCI_df[idx2:]
print(len(EMCI_test_df))


# In[18]:


LMCI_df = pd.read_csv('./mmse/LMCI_APOE.csv')
idx = round(len(LMCI_df)*0.8*0.8)
idx2 = round(len(LMCI_df)*0.8)
LMCI_train_df = LMCI_df[:idx]
print(len(LMCI_train_df))

LMCI_val_df = LMCI_df[idx:idx2]
print(len(LMCI_val_df))

LMCI_test_df = LMCI_df[idx2:]
print(len(LMCI_test_df))


# In[19]:


import pandas as pd
AD_df = pd.read_csv('./mmse/AD_APOE.csv')
idx = round(len(AD_df)*0.8*0.8)
idx2 = round(len(AD_df)*0.8)
AD_train_df = AD_df[:idx]
print(len(AD_train_df))
AD_val_df= AD_df[idx:idx2]
print(len(AD_val_df))

AD_test_df = AD_df[idx2:]
print(len(AD_test_df))


# In[20]:


del(AD_df)
del(CN_df)
del(EMCI_df)
del(LMCI_df)


# In[21]:


train_df = pd.concat([CN_train_df, EMCI_train_df, LMCI_train_df, AD_train_df])
val_df = pd.concat([CN_val_df, EMCI_val_df, LMCI_val_df, AD_val_df])
test_df = pd.concat([CN_test_df, EMCI_test_df, LMCI_test_df, AD_test_df])


# # Dataloader

# In[22]:


class Stacking_Dataset(Dataset):
    def __init__(self, df, image=None, label=None):
        self.df = df
        self.image = image
        self.label = label
#         self.train = train
    def __getitem__(self,index):
        sex = self.df.loc[:,["Sex"]].iloc[index].values
        if sex == "M":
            sex_tensor = torch.FloatTensor([0])
        else:
            sex_tensor = torch.FloatTensor([1])
#         Y = self.df.loc[:,["Label"]].iloc[index].values
        image = self.image[index]
        image_tensor = image.detach().clone()
            
        label_tensor = self.label[index].detach().clone()
#         Y = torch.tensor(Y)
        return sex_tensor, image_tensor, label_tensor
    def __len__(self):
          return len(self.image)
train_loader = DataLoader(Stacking_Dataset(train_df, train_image.float(), train_label.long()), batch_size=16, shuffle=True)
val_loader = DataLoader(Stacking_Dataset(val_df, val_image.float(), val_label.long()), batch_size=8, shuffle = False)


# In[23]:


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
    return weight_per_class
make_weights_for_balanced_classes([int(i) for i in train_label],4)
# train_weights = make_weights_for_balanced_classes([int(i) for i in train_label],4)
# train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights,len(train_weights), replacement=True) 


# In[24]:


# batch_size = 16
# train_loader = DataLoader(TensorDataset(train_image.float(), train_label.long()), batch_size=batch_size, shuffle = True)
# val_loader = DataLoader(TensorDataset(val_image.float(), val_label.long()), batch_size=batch_size, shuffle = False)


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


# In[28]:


def visualize_augmentations(data):
      figure, ax = plt.subplots(nrows = 1, ncols = 5, figsize = (12, 6))
      for i in range(5):
          image= data[i]
          ax.ravel()[i].imshow(image[50,:,:])
          ax.ravel()[i].set_axis_off()
      plt.tight_layout()
      plt.show() 


# In[29]:


class FocalLoss(nn.modules.loss._WeightedLoss):
  def __init__(self, weight = None, gamma = 2, reduction = 'mean'):
        super(FocalLoss, self).__init__(weight, reduction = reduction)
        self.gamma = gamma
        self.weight = weight
  def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction = self.reduction, weight = self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


# In[30]:


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


# # Train

# In[31]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)


# In[32]:


# !pip uninstall torchvision -y


# In[33]:


# !pip install torchvision


# In[34]:


import torchvision
print(torchvision.__version__)
#mc3_18,r2plus1d_18


# In[35]:


class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model, self).__init__()
        self.cnn = models.video.r3d_18(pretrained=True)
        self.features = nn.Sequential(*list(self.cnn.children())[:-1])
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(nn.Linear(num_ftrs, 128),
                                    nn.BatchNorm1d(128),
                                    nn.SELU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(128, 4))
    def forward(self, x): # text = [batch size,sent_length]
 
        logits = self.features(x)
        
        return logits


# In[36]:


class Classifier_model(nn.Module):
    def __init__(self):
        super(Classifier_model, self).__init__()
        num_ftrs = 512
        self.Flatten = torch.nn.Flatten()
        self.fc_male =  nn.Sequential(nn.Linear(num_ftrs, 128),
                                    nn.BatchNorm1d(128),
                                    nn.SELU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(128, 4) 
                                    )
        self.fc_female =  nn.Sequential(nn.Linear(num_ftrs, 128),
                            nn.BatchNorm1d(128),
                            nn.SELU(),
                            nn.Dropout(0.2),
                            nn.Linear(128, 4) 
                            )
    def forward(self, x, sex): # text = [batch size,sent_length]
  
        logits = x
        m = torch.FloatTensor([1]).to(device)-sex
        f = sex
        logits = self.Flatten(logits)
        logits = m * self.fc_male(logits) + f * self.fc_female(logits)
        return logits


# In[37]:


# set seed to get the same results
seed = 456
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# In[38]:


if __name__ == '__main__':
    
    criterion = FocalLoss(weight=torch.FloatTensor([2.4255492270138324, 3.851421188630491, 7.397022332506204, 5.184347826086957]).to(device))
#     criterion = nn.CrossEntropyLoss()
    model = CNN_model().to(device)
    classifier = Classifier_model().to(device)
    PATH = './mmse/300epochs/weight_sex_learningcurves_300.pkl'
    model.load_state_dict(torch.load('./weights/300epochs/weight_4class_noswap_fc_affine_learningcurves_300.pkl'))
    optimizer = optim.Adam(model.parameters(), lr = 5e-5, weight_decay=3e-5)
    loss_resnet18_pre = []
    val_loss_resnet18_pre = []

    train_acc_resnet18_pre = []
    val_acc_resnet18_pre = []
    
    num_epochs = 300
    max_acc = 0.0  
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
    #     scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.9)
        model.train()
        classifier.train()
        for i, (apoe, data, label) in enumerate(train_loader):

          optimizer.zero_grad()
          data = augmentation(data)
            
          data = dim1_to_dim3(data)
          outputs = model(data.to(device)) 
          outputs = classifier(outputs.to(device), apoe.to(device)) 

          # loss
          loss = criterion(outputs, label.to(device))
          train_loss += loss.item()    
          # accuracy
          train_acc += np.sum(np.argmax(outputs.cpu().data.numpy(), axis=1) == label.numpy())
          # update
          loss.backward()
          optimizer.step()
        
        
        model.eval()
        classifier.eval()
        with torch.no_grad():
          val_loss = 0.0
          val_acc = 0.0
          for i, (apoe, data, label) in enumerate(val_loader):

            # forward
            data = dim1_to_dim3(data)
            outputs = model(data.to(device)) 
            outputs = classifier(outputs.to(device), apoe.to(device)) 
            loss = criterion(outputs, label.to(device))
            val_loss += loss.item()

            val_acc += np.sum(np.argmax(outputs.cpu().data.numpy(), axis=1) == label.numpy())

        if val_acc/len(val_image) > max_acc:
            max_acc = val_acc/len(val_image)
            torch.save(model.state_dict(), PATH)
            torch.save(classifier.state_dict(), './mmse/300epochs/weight_sex_learningcurves_classifier_300.pkl')
        loss_resnet18_pre.append(train_loss/len(train_image))
        val_loss_resnet18_pre.append(val_loss/len(val_image)) 
        train_acc_resnet18_pre.append(train_acc/len(train_image))
        val_acc_resnet18_pre.append(val_acc/len(val_image))
                       
        if epoch == 149:  
            torch.save(model.state_dict(), './mmse/300epochs/weight_sex_learningcurves_150.pkl')
            torch.save(classifier.state_dict(), './mmse/300epochs/weight_sex_learningcurves_classifier_150.pkl')

        if (epoch+1) % 1 == 0:
          print('Epoch[{}/{}],Loss:{:.4f},Train Accuracy:{:.2f},Valid Loss:{:.4f},Valid Accuracy:{:.2f}'
          .format(epoch+1,num_epochs,train_loss/len(train_image),train_acc/len(train_image),val_loss/len(val_image),val_acc/len(val_image)))


# In[39]:


history = {"train_loss": loss_resnet18_pre,
          "val_loss": val_loss_resnet18_pre,
          "train_acc": train_acc_resnet18_pre,
          "val_acc": val_acc_resnet18_pre}


# In[40]:


pd.DataFrame(history).to_csv("./mmse/300epochs/history_sex_learningcurves.csv")


# In[41]:


plt.plot(loss_resnet18_pre, label='train')
plt.plot(val_loss_resnet18_pre, label='val')
plt.legend(loc='upper right')
plt.title("Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()


# In[42]:


plt.plot(train_acc_resnet18_pre, label='train_acc')
plt.plot(val_acc_resnet18_pre, label='val_acc')
plt.plot(np.array(loss_resnet18_pre)*10, label='train_loss')
plt.legend(loc='upper left')
plt.title("Accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()


# # Test

# In[46]:


model = CNN_model().to(device)
classifier = Classifier_model().to(device)
model.load_state_dict(torch.load(PATH))
classifier.load_state_dict(torch.load('./mmse/300epochs/weight_sex_learningcurves_classifier_300.pkl'))


# In[47]:


test_loader = DataLoader(Stacking_Dataset(test_df, test_image.float(), test_label.long()), batch_size=8, shuffle = False)


# In[48]:


pred_label_pre = []
test_acc = 0.0
model.eval()
classifier.eval()
for i, (apoe, data, label) in enumerate(test_loader):

      # forward
      data = dim1_to_dim3(data)
#       outputs = model(data.to(device))
      outputs = model(data.to(device)) 
      outputs = classifier(outputs.to(device), apoe.to(device)) 
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


# ### Confusion Matrix

# In[49]:


confmat = confusion_matrix(y_true=test_label, y_pred=pred_label_pre)

df_cm = pd.DataFrame(confmat, index = [i for i in ["CN", "EMCI", "LMCI", "AD"]],
                     columns = [i for i in ["CN", "EMCI", "LMCI", "AD"]])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g', annot_kws={"size":20})
plt.xlabel('predicted label')        
plt.ylabel('true label')
plt.title('Confusion Matrix')
plt.show()


# In[50]:


#Sensitivity
Class = ["CN", "EMCI", "LMCI", "AD"]
for i in range(4):
  print(Class[i], end=": ")
  print(confmat[i][i]/sum(confmat[i]))


# In[51]:


#Specificity
for i in range(4):
  print(Class[i], end=": ")
  print((np.trace(confmat) - confmat[i][i])/(sum(sum(confmat)) - sum(confmat[i])))


# # ROC Curve

# In[52]:


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


# In[53]:


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


# In[54]:


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


# # Baseline (zero rule algorithm classification)

# In[47]:


from random import seed
from random import randrange
# zero rule algorithm for classification
def zero_rule_algorithm_classification(train, test):
    output_values = [row[-1] for row in train]
    prediction = max(set(output_values), key=output_values.count)
    predicted = [prediction for i in range(len(test))]
    return predicted


# In[53]:


seed(1)

train = [['0']]*len(CN_image) + [['1']] * len(EMCI_image) + [['2']] * len(LMCI_image) + [['3']] * len(AD_image)
test = [[None]]*len(test_label)

predictions = zero_rule_algorithm_classification(train, test)
count = 0
for i,j in zip(test_label.tolist(), predictions):
    if int(i) == int(j):
        count += 1
print(count / len(test))


# In[58]:


predictions = [int(i) for i in predictions]


# In[59]:


confmat = confusion_matrix(y_true=test_label, y_pred=predictions)

df_cm = pd.DataFrame(confmat, index = [i for i in ["CN", "EMCI", "LMCI", "AD"]],
                     columns = [i for i in ["CN", "EMCI", "LMCI", "AD"]])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
plt.xlabel('predicted label')        
plt.ylabel('true label')
plt.title('Confusion Matrix')
plt.show()


# In[60]:


#Sensitivity
Class = ["CN", "EMCI", "LMCI", "AD"]
for i in range(4):
  print(Class[i], end=": ")
  print(confmat[i][i]/sum(confmat[i]))


# In[61]:


#Specificity
for i in range(4):
  print(Class[i], end=": ")
  print((np.trace(confmat) - confmat[i][i])/(sum(sum(confmat)) - sum(confmat[i])))


# # Baseline (random baseline)

# In[62]:


# Generate random predictions
def random_algorithm(train, test):
    output_values = [row[-1] for row in train]
    unique = list(set(output_values))
    predicted = list()
    for row in test:
        index = randrange(len(unique))
        predicted.append(unique[index])
    return predicted


# In[64]:


predictions = random_algorithm(train, test)
count = 0
for i,j in zip(test_label.tolist(), predictions):
    if int(i) == int(j):
        count += 1
print(count / len(test))


# In[66]:


predictions = [int(i) for i in predictions]


# In[67]:


confmat = confusion_matrix(y_true=test_label, y_pred=predictions)

df_cm = pd.DataFrame(confmat, index = [i for i in ["CN", "EMCI", "LMCI", "AD"]],
                     columns = [i for i in ["CN", "EMCI", "LMCI", "AD"]])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
plt.xlabel('predicted label')        
plt.ylabel('true label')
plt.title('Confusion Matrix')
plt.show()


# In[68]:


#Sensitivity
Class = ["CN", "EMCI", "LMCI", "AD"]
for i in range(4):
  print(Class[i], end=": ")
  print(confmat[i][i]/sum(confmat[i]))


# In[69]:


#Specificity
for i in range(4):
  print(Class[i], end=": ")
  print((np.trace(confmat) - confmat[i][i])/(sum(sum(confmat)) - sum(confmat[i])))


# In[ ]:




