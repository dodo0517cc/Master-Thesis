# Master-Thesis

## Introduction



## Objective

Detect the phase of Alzheimer’s Disease from MRI images by multiclass classification techniques

## Data preprocessing

1. Brain extraction

2. Tissue segmentation

3. Resample images

4. Crop or Pad

5. Min-max normalization

6. Remove useless pixels

7. Augmentation (Random Horizontal Flip / Random Noise / Random Bias Field / Random Affine)

<img width="500" alt="image" src="https://user-images.githubusercontent.com/77607182/178153903-cef28e62-ea64-42e3-9b03-19260a67bcfb.png">


## Pre-trained MODEL: 

ResNet3D-18

## Methodology: 

1. Direct multiclass classification

2. Hierarchical classification

3. One-vs-one clssification

4. One-vs-all classification

5. Ensemble learning

6. Add clinical data with MRIs


## Results

#### MRIs:

Accuracy: 0.950

Recall: 0.937

Specificity: 0.948

#### MRIs with APOE (Apolipoprotein E): 

Accuracy: 0.952

Recall: 0.941

Specificity: 0.950


