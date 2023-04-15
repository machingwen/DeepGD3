# -*- coding: utf-8 -*-
from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
import argparse
import time
import math
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from util import *
from torchvision import transforms, datasets
from networks.mobilenetv3 import SupConMobileNetV3Large
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pickle
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc
from numpy import interp
from matplotlib import pyplot as plt
from torch.distributions import multivariate_normal
from pycave.bayes.gmm import GaussianMixture
from pycave.bayes.gmm.model import GaussianMixtureModel

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed for datasets')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='10,20,25',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='mobilenetv3_large')
    parser.add_argument('--dataset', type=str, default='phison',
                        choices=['cifar10', 'cifar100', 'phison'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=224, help='parameter for RandomResizedCrop')

    # other setting
    parser.add_argument('--split', type=str, default="val", help='val or test component for cosine similarity calculation')
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to final model')
    parser.add_argument('--componentName', type=int, default=0,
                        help='name of test component')
    parser.add_argument("--embedding_layer", type=str, default="shared_embedding", help="Which embedding to visualization( encoder or shared_embedding)")
    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models_classifier'.format(opt.dataset)

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'phison':
        opt.n_cls = 2
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    return opt

def set_loader(opt):
    
    transform = transforms.Compose([
        transforms.Resize([opt.size, opt.size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    if opt.split == "val":
        df = CreateDataset_relabel_for_val_component(opt.seed, component_name=opt.componentName)
        if opt.componentName == 21 or opt.componentName == 22:
            df = df
        else:
            df = df.loc[(df['class'] == 0)]
            
    if opt.split == "test":
        df = CreateDataset_relabel_for_test_component(opt.seed, component_name=opt.componentName)
        
    dataset = CustomDataset(df, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)
    return loader

def set_model(opt):
    model = SupConMobileNetV3Large(name=opt.model)
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict_model = ckpt['model']
    
    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True
        model.load_state_dict(state_dict_model)

    return model

def load_GMM_model(opt):
    gmm=GaussianMixture()

    GMM_0_attribute  = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_0_Classifier", "model"))
    GMM_1_attribute  = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_1_Classifier", "model"))
    GMM_2_attribute  = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_2_Classifier", "model"))
    GMM_3_attribute  = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_3_Classifier", "model"))
    GMM_4_attribute  = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_4_Classifier", "model"))
    GMM_5_attribute  = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_5_Classifier", "model"))
    GMM_6_attribute  = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_6_Classifier", "model"))
    GMM_7_attribute  = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_7_Classifier", "model"))
    GMM_8_attribute  = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_8_Classifier", "model"))
    GMM_9_attribute  = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_9_Classifier", "model"))
    GMM_10_attribute = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_10_Classifier", "model"))
    GMM_11_attribute = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_11_Classifier", "model"))
    GMM_12_attribute = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_12_Classifier", "model"))
    GMM_13_attribute = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_13_Classifier", "model"))
    GMM_14_attribute = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_14_Classifier", "model"))
    GMM_15_attribute = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_15_Classifier", "model"))
    GMM_16_attribute = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_16_Classifier", "model"))
    GMM_17_attribute = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_17_Classifier", "model"))
    GMM_18_attribute = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_18_Classifier", "model"))
    GMM_19_attribute = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_19_Classifier", "model"))
    GMM_20_attribute = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_20_Classifier", "model"))
    GMM_21_attribute = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_21_Classifier", "model"))
    GMM_22_attribute = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_22_Classifier", "model"))
    
    GMM_0 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_0_Classifier")
    GMM_1 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_1_Classifier")
    GMM_2 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_2_Classifier")
    GMM_3 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_3_Classifier")
    GMM_4 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_4_Classifier")
    GMM_5 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_5_Classifier")
    GMM_6 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_6_Classifier")
    GMM_7 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_7_Classifier")
    GMM_8 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_8_Classifier")
    GMM_9 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_9_Classifier")
    GMM_10 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_10_Classifier")
    GMM_11 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_11_Classifier")
    GMM_12 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_12_Classifier")
    GMM_13 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_13_Classifier")
    GMM_14 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_14_Classifier")
    GMM_15 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_15_Classifier")
    GMM_16 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_16_Classifier")
    GMM_17 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_17_Classifier")
    GMM_18 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_18_Classifier")
    GMM_19 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_19_Classifier")
    GMM_20 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_20_Classifier")
    GMM_21 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_21_Classifier")
    GMM_22 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+/dataset_{opt.seed}_GMM_22_Classifier")
    return GMM_0_attribute,GMM_1_attribute,GMM_2_attribute,GMM_3_attribute,GMM_4_attribute,GMM_5_attribute,GMM_6_attribute,GMM_7_attribute,GMM_8_attribute,GMM_9_attribute,GMM_10_attribute,GMM_11_attribute,GMM_12_attribute,GMM_13_attribute,GMM_14_attribute,GMM_15_attribute,GMM_16_attribute,GMM_17_attribute,GMM_18_attribute,GMM_19_attribute,GMM_20_attribute,GMM_21_attribute,GMM_22_attribute, GMM_0,GMM_1,GMM_2,GMM_3,GMM_4,GMM_5,GMM_6,GMM_7,GMM_8,GMM_9,GMM_10,GMM_11,GMM_12,GMM_13,GMM_14,GMM_15,GMM_16,GMM_17,GMM_18,GMM_19,GMM_20,GMM_21,GMM_22


def calculateCosineSimilarity(opt, features_mean,
                              GMM_0_attribute,GMM_1_attribute,GMM_2_attribute,
                              GMM_3_attribute,GMM_4_attribute,GMM_5_attribute,
                              GMM_6_attribute,GMM_7_attribute,GMM_8_attribute,
                              GMM_9_attribute,GMM_10_attribute,GMM_11_attribute,
                              GMM_12_attribute,GMM_13_attribute,GMM_14_attribute,
                              GMM_15_attribute,GMM_16_attribute,GMM_17_attribute,
                              GMM_18_attribute,GMM_19_attribute,GMM_20_attribute,
                              GMM_21_attribute,GMM_22_attribute
                             ):
    cos = nn.CosineSimilarity(dim=1)
    similarity_0 = cos(torch.mean(GMM_0_attribute.means, dim=0, keepdim=True), features_mean).item()
    similarity_1 = cos(torch.mean(GMM_1_attribute.means, dim=0, keepdim=True), features_mean).item()
    similarity_2 = cos(torch.mean(GMM_2_attribute.means, dim=0, keepdim=True), features_mean).item()
    similarity_3 = cos(torch.mean(GMM_3_attribute.means, dim=0, keepdim=True), features_mean).item()
    similarity_4 = cos(torch.mean(GMM_4_attribute.means, dim=0, keepdim=True), features_mean).item()
    similarity_5 = cos(torch.mean(GMM_5_attribute.means, dim=0, keepdim=True), features_mean).item()
    similarity_6 = cos(torch.mean(GMM_6_attribute.means, dim=0, keepdim=True), features_mean).item()
    similarity_7 = cos(torch.mean(GMM_7_attribute.means, dim=0, keepdim=True), features_mean).item()
    similarity_8 = cos(torch.mean(GMM_8_attribute.means, dim=0, keepdim=True), features_mean).item()
    similarity_9 = cos(torch.mean(GMM_9_attribute.means, dim=0, keepdim=True), features_mean).item()
    similarity_10 = cos(torch.mean(GMM_10_attribute.means, dim=0, keepdim=True), features_mean).item()
    similarity_11 = cos(torch.mean(GMM_11_attribute.means, dim=0, keepdim=True), features_mean).item()
    similarity_12 = cos(torch.mean(GMM_12_attribute.means, dim=0, keepdim=True), features_mean).item()
    similarity_13 = cos(torch.mean(GMM_13_attribute.means, dim=0, keepdim=True), features_mean).item()
    similarity_14 = cos(torch.mean(GMM_14_attribute.means, dim=0, keepdim=True), features_mean).item()
    similarity_15 = cos(torch.mean(GMM_15_attribute.means, dim=0, keepdim=True), features_mean).item()
    similarity_16 = cos(torch.mean(GMM_16_attribute.means, dim=0, keepdim=True), features_mean).item()
    similarity_17 = cos(torch.mean(GMM_17_attribute.means, dim=0, keepdim=True), features_mean).item()
    similarity_18 = cos(torch.mean(GMM_18_attribute.means, dim=0, keepdim=True), features_mean).item()
    similarity_19 = cos(torch.mean(GMM_19_attribute.means, dim=0, keepdim=True), features_mean).item()
    similarity_20 = cos(torch.mean(GMM_20_attribute.means, dim=0, keepdim=True), features_mean).item()
    similarity_21 = cos(torch.mean(GMM_21_attribute.means, dim=0, keepdim=True), features_mean).item()
    similarity_22 = cos(torch.mean(GMM_22_attribute.means, dim=0, keepdim=True), features_mean).item()    
    
    data = {'TrainComponentName': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        'CosineSimilarity': [similarity_0, similarity_1, similarity_2, similarity_3, similarity_4, similarity_5, similarity_6, similarity_7, similarity_8, similarity_9, similarity_10, similarity_11, similarity_12, similarity_13, similarity_14, similarity_15, similarity_16, similarity_17, similarity_18, similarity_19, similarity_20, similarity_21, similarity_22]}
  
    df = pd.DataFrame(data)    
    df.to_csv(f'./output/{opt.seed}/{opt.split}_{opt.componentName}_cosine_similarity.csv', index=False)
    
def main():
    opt = parse_option()    
    dataLoader = set_loader(opt)
    model = set_model(opt)  
    GMM_0_attribute,GMM_1_attribute,GMM_2_attribute,GMM_3_attribute,GMM_4_attribute,GMM_5_attribute,GMM_6_attribute,GMM_7_attribute,GMM_8_attribute,GMM_9_attribute,GMM_10_attribute,GMM_11_attribute,GMM_12_attribute,GMM_13_attribute,GMM_14_attribute,GMM_15_attribute,GMM_16_attribute,GMM_17_attribute,GMM_18_attribute,GMM_19_attribute,GMM_20_attribute,GMM_21_attribute,GMM_22_attribute, GMM_0,GMM_1,GMM_2,GMM_3,GMM_4,GMM_5,GMM_6,GMM_7,GMM_8,GMM_9,GMM_10,GMM_11,GMM_12,GMM_13,GMM_14,GMM_15,GMM_16,GMM_17,GMM_18,GMM_19,GMM_20,GMM_21,GMM_22 = load_GMM_model(opt)
    
    
    features, gt_labels, image_paths, name_label_list, _, _ = get_features_trained_weight(model, dataLoader, opt.embedding_layer)
    features_mean = torch.mean(torch.from_numpy(features), dim=0, keepdim=True)
    
    calculateCosineSimilarity(opt, features_mean,
                              GMM_0_attribute,GMM_1_attribute,GMM_2_attribute,
                              GMM_3_attribute,GMM_4_attribute,GMM_5_attribute,
                              GMM_6_attribute,GMM_7_attribute,GMM_8_attribute,
                              GMM_9_attribute,GMM_10_attribute,GMM_11_attribute,
                              GMM_12_attribute,GMM_13_attribute,GMM_14_attribute,
                              GMM_15_attribute,GMM_16_attribute,GMM_17_attribute,
                              GMM_18_attribute,GMM_19_attribute,GMM_20_attribute,
                              GMM_21_attribute,GMM_22_attribute
                             )
if __name__ == '__main__':
    main()
