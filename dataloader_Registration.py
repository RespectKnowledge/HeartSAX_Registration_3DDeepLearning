# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 16:29:13 2023

@author: aq22
"""
#%% kfold #####################
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

#%% Dataloader 
import os
import torch
from torch.utils.data import Dataset,DataLoader
import SimpleITK as sitk
import pandas as pd
from skimage.transform import resize
class regsax(Dataset):

  def __init__(self,pathcsv,pathlabel,rsize):
    self.pathcsv=pathcsv
    self.pathlabel=pathlabel
    #self.pathdata=pathdata
    self.rsize=rsize
    self.df=pd.read_csv(self.pathcsv)



  def __getitem__(self,index):
    pathfixed=self.df['fixed'][index]
    pathmove=pathfixed.replace('ED','ES')

    fixed_img=sitk.ReadImage(pathfixed)
    move_img=sitk.ReadImage(pathmove)
    fixed_array=sitk.GetArrayFromImage(fixed_img).swapaxes(2, 0)
    moved_array=sitk.GetArrayFromImage(move_img).swapaxes(2, 0)
    #print(fixed_img.GetSize())
    #print(move_img.GetSize())
    fixed_array_resize=resize(fixed_array, output_shape=(self.rsize[0],self.rsize[1],self.rsize[2]),preserve_range=True, anti_aliasing=False,order=3)
    move_array_resize=resize(moved_array, output_shape=(self.rsize[0],self.rsize[1],self.rsize[2]),preserve_range=True, anti_aliasing=False,order=3)
    fixedimg=torch.from_numpy(fixed_array_resize).unsqueeze(0)
    moveimg=torch.from_numpy(move_array_resize).unsqueeze(0)

    ################## read labels ###################
    fixed_label=os.path.join(self.pathlabel,pathfixed.split('/')[-1].replace('ED','ED_label_gt'))
    move_label=os.path.join(self.pathlabel,pathmove.split('/')[-1].replace('ES','ES_label_gt'))

    fixed_labl=sitk.ReadImage(fixed_label)
    move_labl=sitk.ReadImage(move_label)
    fixed_arrayla=sitk.GetArrayFromImage(fixed_labl).swapaxes(2, 0)
    moved_arrayla=sitk.GetArrayFromImage(move_labl).swapaxes(2, 0)
    #print(fixed_img.GetSize())
    #print(move_img.GetSize())
    fixed_array_resize_la=resize(fixed_arrayla, output_shape=(self.rsize[0],self.rsize[1],self.rsize[2]),preserve_range=True, anti_aliasing=False,order=0)
    move_array_resize_la=resize(moved_arrayla, output_shape=(self.rsize[0],self.rsize[1],self.rsize[2]),preserve_range=True, anti_aliasing=False,order=0)
    fixedlabel=torch.from_numpy(fixed_array_resize_la).unsqueeze(0)
    movelabel=torch.from_numpy(move_array_resize_la).unsqueeze(0)


    return fixedimg,moveimg,fixedlabel,movelabel

  def __len__(self):
    return len(self.df['fixed'])

path='/home/aqayyum/SAXregistration/kfold/train_0.csv'
pathlab='/home/aqayyum/SAXregistration/labels'
dataset_train=regsax(path,pathlab,(224,224,32))
#img1,img2,lab1m,lab2m=datatrain[0]
#print(img1.shape)
#print(img2.shape)

# train_loader=DataLoader(datatrain,batch_size=1,shuffle=True)
# for i,d in enumerate(train_loader):
#   im1,im2,lab1,lab2=d
#   print(im1.shape)
#   print(im2.shape)
#   break

path='/home/aqayyum/SAXregistration/kfold/val_0.csv'
pathlab='/home/aqayyum/SAXregistration/labels'
dataset_valid=regsax(path,pathlab,(224,224,32))
#img1,img2,lab1m,lab2m=datatrain[0]


import enum
from torch.utils.data import DataLoader
train_loader=DataLoader(dataset_train,batch_size=1,shuffle=True,num_workers=4,pin_memory=True)
# for i,d in enumerate(train_data):
#   img,lab=d
#   print(img.shape)
#   print(lab.shape)
#   break
val_loader=DataLoader(dataset_valid,batch_size=1,shuffle=False,num_workers=4,pin_memory=True)
