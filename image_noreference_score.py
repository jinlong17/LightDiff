import torch
import pyiqa
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



######envirnment   py3.7-torch1.8
#####
print(pyiqa.list_models())

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


###### PATHs
path = '../CAM_FRONT'


###
iqa_metric = pyiqa.create_metric('musiq',device=device)

print(iqa_metric.lower_better)


musiq_socres = []
for img in os.listdir(path):
    ful_path = os.path.join(path,img)
    socre = iqa_metric(ful_path)
    musiq_socres.append(socre.item())
    #print(socre.item())
musiq_mean = sum(musiq_socres) / len(musiq_socres)
print('musiq:',musiq_mean)

###
iqa_metric = pyiqa.create_metric('niqe',device=device)

print(iqa_metric.lower_better)


niqe_socres = []
for img in os.listdir(path):
    ful_path = os.path.join(path,img)
    socre = iqa_metric(ful_path)
    niqe_socres.append(socre.item())
    #print(socre.item())
niqe_mean = sum(niqe_socres) / len(niqe_socres)
print('niqe:',niqe_mean)

###hyperiqa  cvpr2020 cition 309
iqa_metric = pyiqa.create_metric('hyperiqa',device=device)
print(iqa_metric.lower_better)
hyperiqa_socres = []
for img in os.listdir(path):
    ful_path = os.path.join(path,img)
    socre = iqa_metric(ful_path)
    hyperiqa_socres.append(socre.item())
    #print(socre.item())
hyperiqa_mean = sum(hyperiqa_socres) / len(hyperiqa_socres)
print('hyperiqa:',hyperiqa_mean)


###ilniqe  tip2015  cition 948
iqa_metric = pyiqa.create_metric('ilniqe',device=device)
print(iqa_metric.lower_better)
ilniqe_socres = []
for img in os.listdir(path):
    ful_path = os.path.join(path,img)
    socre = iqa_metric(ful_path)
    ilniqe_socres.append(socre.item())
    #print(socre.item())
ilniqe_mean = sum(ilniqe_socres) / len(ilniqe_socres)
print('ilniqe:',ilniqe_mean)

###maniqa  cvprw2022 cition 67
iqa_metric = pyiqa.create_metric('maniqa',device=device)
print(iqa_metric.lower_better)
maniqa_socres = []
for img in os.listdir(path):
    ful_path = os.path.join(path,img)
    socre = iqa_metric(ful_path)
    maniqa_socres.append(socre.item())
    #print(socre.item())
maniqa_mean = sum(maniqa_socres) / len(maniqa_socres)
print('maniqa:',maniqa_mean)

###nima  tip2018  cition 793
iqa_metric = pyiqa.create_metric('nima',device=device)
print(iqa_metric.lower_better)
nima_socres = []
for img in os.listdir(path):
    ful_path = os.path.join(path,img)
    socre = iqa_metric(ful_path)
    nima_socres.append(socre.item())
    #print(socre.item())
nima_mean = sum(nima_socres) / len(nima_socres)
print('nima:',nima_mean)

###tres  wacv2022  cition 92
iqa_metric = pyiqa.create_metric('tres',device=device)
print(iqa_metric.lower_better)
tres_socres = []
for img in os.listdir(path):
    ful_path = os.path.join(path,img)
    socre = iqa_metric(ful_path)
    tres_socres.append(socre.item())
    #print(socre.item())
tres_mean = sum(tres_socres) / len(tres_socres)
print('tres:',tres_mean)