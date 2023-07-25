# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 11:40:06 2023

@author: aq22
"""

import os
from glob import glob
from skimage.transform import resize
import random
#resize(image, output_shape=shape)
path='/home/aqayyum/SAXregistration'
pathdataED=sorted(glob(os.path.join(path,'images','*_ED.nii.gz')))
pathdataES=sorted(glob(os.path.join(path,'images','*_ES.nii.gz')))
#pathdataED_lab=glob(os.path.join(path,'labels','*_ED_label_gt.nii.gz'))
pathsave='/home/aqayyum/SAXregistration/kfold'
seed=0
k_fold=5
import pandas as pd
for k in range(0,5):
  #lstdata=subj
  random.Random(seed).shuffle(pathdataED)
  length=len(pathdataED)
  print(length)
  print(k)
  #break
  test_list=pathdataED[k*(length//k_fold):(k+1)*(length//k_fold)]
  trainlst=list(set(pathdataED)-set(test_list))
  trainlstdf=pd.DataFrame(trainlst,columns=['fixed'])
  test_listdf=pd.DataFrame(test_list,columns=['fixed'])
  trainlstdf.to_csv(os.path.join(pathsave,'train_'+str(k)+'.csv'),index=False)
  test_listdf.to_csv(os.path.join(pathsave,'val_'+str(k)+'.csv'),index=False)
  #break