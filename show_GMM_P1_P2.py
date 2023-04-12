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
    parser.add_argument('--relabel', action='store_true',
                        help='relabel dataset')
    parser.add_argument('--std_1', type=float, default=0.1,
                            help='momentum')
    parser.add_argument('--std_3', type=float, default=0.3,
                            help='momentum')
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to final model')
    parser.add_argument('--TestComponent', type=str, default='',
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
    # construct data loader
    if opt.dataset == 'phison':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    val_transform = transforms.Compose([
        transforms.Resize([opt.size, opt.size]),
        transforms.ToTensor(),
        normalize,
    ])
    if opt.dataset == 'phison':
        train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, _ = CreateDataset_relabel(opt.seed, testing=True)
        test_df_mapping2_label = test_df.copy()    
        name_of_each_component = test_df_mapping2_label['component_name'].value_counts().index.tolist()
        num_of_image_in_each_component = test_df_mapping2_label['component_name'].value_counts().values
        test_component_name_df = pd.DataFrame(list(zip(name_of_each_component, num_of_image_in_each_component)), columns =['component_name', 'total'])

        for name in set(test_df_mapping2_label['component_name'].values):
            temp_data = test_df_mapping2_label.loc[(test_df_mapping2_label["component_name"] == name)]
            for k, v in zip(temp_data['class'].value_counts().keys(), temp_data['class'].value_counts()):
                if k == 0:
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'good'] = temp_data['class'].value_counts().sort_index().values[0]
                elif k ==1:
                    try:
                        test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[1]
                    except:
                        print(f"{name} only contains bad label.")
                        test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[0]
        test_component_name_df['good'] = test_component_name_df['good'].fillna(0).astype(int)
        try:
            test_component_name_df['bad'] = test_component_name_df['bad'].fillna(0).astype(int)
            test_component_name_df = test_component_name_df[['component_name', 'total', 'good', 'bad']]    
        except:
            test_component_name_df = test_component_name_df[['component_name', 'total', 'good']]        
        
        col = {'overkill': 0, 'leakage': 0, 'unknown':  0,}
        test_component_name_df = test_component_name_df.assign(**col)

        val_dataset = CustomDataset(val_df, transform=val_transform)
        test_dataset = CustomDataset(test_df, transform=val_transform)
    else:
        raise ValueError(opt.dataset)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)
    
    return val_loader, test_loader, test_component_name_df, train_component_label, val_component_label, test_component_label

def set_model(opt):
    model = SupConMobileNetV3Large(name=opt.model)
        
    criterion = torch.nn.CrossEntropyLoss()

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict_model = ckpt['model']
    
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
        model.load_state_dict(state_dict_model)
    else:
        model.load_state_dict(state_dict_model)

    return model, criterion

def load_GMM_model(opt):
    gmm=GaussianMixture()

    GMM_0_attribute  = GaussianMixtureModel.load(os.path.join(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_0_Classifier", "model"))
    GMM_1_attribute  = GaussianMixtureModel.load(os.path.join(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_1_Classifier", "model"))
    GMM_2_attribute  = GaussianMixtureModel.load(os.path.join(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_2_Classifier", "model"))
    GMM_3_attribute  = GaussianMixtureModel.load(os.path.join(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_3_Classifier", "model"))
    GMM_4_attribute  = GaussianMixtureModel.load(os.path.join(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_4_Classifier", "model"))
    GMM_5_attribute  = GaussianMixtureModel.load(os.path.join(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_5_Classifier", "model"))
    GMM_6_attribute  = GaussianMixtureModel.load(os.path.join(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_6_Classifier", "model"))
    GMM_7_attribute  = GaussianMixtureModel.load(os.path.join(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_7_Classifier", "model"))
    GMM_8_attribute  = GaussianMixtureModel.load(os.path.join(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_8_Classifier", "model"))
    GMM_9_attribute  = GaussianMixtureModel.load(os.path.join(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_9_Classifier", "model"))
    GMM_10_attribute = GaussianMixtureModel.load(os.path.join(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_10_Classifier", "model"))
    GMM_11_attribute = GaussianMixtureModel.load(os.path.join(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_11_Classifier", "model"))
    GMM_12_attribute = GaussianMixtureModel.load(os.path.join(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_12_Classifier", "model"))
    GMM_13_attribute = GaussianMixtureModel.load(os.path.join(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_13_Classifier", "model"))
    GMM_14_attribute = GaussianMixtureModel.load(os.path.join(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_14_Classifier", "model"))
    GMM_15_attribute = GaussianMixtureModel.load(os.path.join(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_15_Classifier", "model"))
    GMM_16_attribute = GaussianMixtureModel.load(os.path.join(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_16_Classifier", "model"))
    GMM_17_attribute = GaussianMixtureModel.load(os.path.join(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_17_Classifier", "model"))
    GMM_18_attribute = GaussianMixtureModel.load(os.path.join(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_18_Classifier", "model"))
    GMM_19_attribute = GaussianMixtureModel.load(os.path.join(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_19_Classifier", "model"))
    GMM_20_attribute = GaussianMixtureModel.load(os.path.join(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_20_Classifier", "model"))
    GMM_21_attribute = GaussianMixtureModel.load(os.path.join(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_21_Classifier", "model"))
    GMM_22_attribute = GaussianMixtureModel.load(os.path.join(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_22_Classifier", "model"))
    
    GMM_0 = gmm.load(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_0_Classifier")
    GMM_1 = gmm.load(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_1_Classifier")
    GMM_2 = gmm.load(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_2_Classifier")
    GMM_3 = gmm.load(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_3_Classifier")
    GMM_4 = gmm.load(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_4_Classifier")
    GMM_5 = gmm.load(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_5_Classifier")
    GMM_6 = gmm.load(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_6_Classifier")
    GMM_7 = gmm.load(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_7_Classifier")
    GMM_8 = gmm.load(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_8_Classifier")
    GMM_9 = gmm.load(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_9_Classifier")
    GMM_10 = gmm.load(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_10_Classifier")
    GMM_11 = gmm.load(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_11_Classifier")
    GMM_12 = gmm.load(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_12_Classifier")
    GMM_13 = gmm.load(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_13_Classifier")
    GMM_14 = gmm.load(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_14_Classifier")
    GMM_15 = gmm.load(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_15_Classifier")
    GMM_16 = gmm.load(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_16_Classifier")
    GMM_17 = gmm.load(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_17_Classifier")
    GMM_18 = gmm.load(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_18_Classifier")
    GMM_19 = gmm.load(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_19_Classifier")
    GMM_20 = gmm.load(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_20_Classifier")
    GMM_21 = gmm.load(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_21_Classifier")
    GMM_22 = gmm.load(f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_22_Classifier")
    return GMM_0_attribute,GMM_1_attribute,GMM_2_attribute,GMM_3_attribute,GMM_4_attribute,GMM_5_attribute,GMM_6_attribute,GMM_7_attribute,GMM_8_attribute,GMM_9_attribute,GMM_10_attribute,GMM_11_attribute,GMM_12_attribute,GMM_13_attribute,GMM_14_attribute,GMM_15_attribute,GMM_16_attribute,GMM_17_attribute,GMM_18_attribute,GMM_19_attribute,GMM_20_attribute,GMM_21_attribute,GMM_22_attribute, GMM_0,GMM_1,GMM_2,GMM_3,GMM_4,GMM_5,GMM_6,GMM_7,GMM_8,GMM_9,GMM_10,GMM_11,GMM_12,GMM_13,GMM_14,GMM_15,GMM_16,GMM_17,GMM_18,GMM_19,GMM_20,GMM_21,GMM_22

def pred(opt, GMM_name, probs1, probs2):    
    probs1 = np.where(probs1.numpy() > 1, 0, probs1)
    probs2 = np.where(probs2.numpy() > 1, 0, probs2)
    P1 = []
    P2 =[]
        
    for ii in range(len(probs1)):
        P1.append(probs1[ii][ii])

    for ii in range(len(probs2)):
        P2.append(probs2[ii][ii])
    df = pd.DataFrame(list(zip(P1, P2)), columns =['P1', 'P2'])
    df.to_csv(f"./output/{opt.seed}/{GMM_name}_P1_P2.csv", index=False)
    return P1, P2

def get_prob(X, k, mu, cov, phi, factor=0.1):    
    X = X.cuda()
    mu = mu.cuda()
    cov = cov.cuda()
    likelihood = torch.zeros(X.shape[0], k)
    for i in tqdm(range(k)):
        distribution = multivariate_normal.MultivariateNormal(loc=mu[i], covariance_matrix=cov[i]*factor*factor)
        likelihood[:,i] = distribution.log_prob(X)

    numerator = likelihood * phi                         
    denominator = numerator.sum(axis=1)[:, np.newaxis] 
    weights = numerator / (denominator)
    return torch.exp(weights)

def GMM_Predict_prob(opt, GMM_0_attribute, GMM_1_attribute, GMM_2_attribute, GMM_3_attribute, GMM_4_attribute, GMM_5_attribute, GMM_6_attribute, GMM_7_attribute, GMM_8_attribute, GMM_9_attribute, GMM_10_attribute, GMM_11_attribute, GMM_12_attribute, GMM_13_attribute, GMM_14_attribute, GMM_15_attribute, GMM_16_attribute, GMM_17_attribute, GMM_18_attribute, GMM_19_attribute, GMM_20_attribute, GMM_21_attribute, GMM_22_attribute, GMM_0,GMM_1,GMM_2,GMM_3,GMM_4,GMM_5,GMM_6,GMM_7,GMM_8,GMM_9,GMM_10,GMM_11,GMM_12,GMM_13,GMM_14,GMM_15,GMM_16,GMM_17,GMM_18,GMM_19,GMM_20,GMM_21,GMM_22):
    
    df = pd.read_json(f"{opt.seed}_tau1_tau2_logs.json",lines=True)
    std_threshold_dict = df.iloc[df['target'].idxmax()].tolist()[1]
    
    print('compute probability P1')
    Island_0_th_pred= get_prob(GMM_0_attribute.means, GMM_0.num_components, GMM_0_attribute.means, GMM_0_attribute.covariances, GMM_0_attribute.component_probs, factor=std_threshold_dict['tau1']).squeeze()
    Island_1_th_pred= get_prob(GMM_1_attribute.means, GMM_1.num_components, GMM_1_attribute.means, GMM_1_attribute.covariances, GMM_1_attribute.component_probs, factor=std_threshold_dict['tau1']).squeeze()
    Island_2_th_pred= get_prob(GMM_2_attribute.means, GMM_2.num_components, GMM_2_attribute.means, GMM_2_attribute.covariances, GMM_2_attribute.component_probs, factor=std_threshold_dict['tau1']).squeeze()
    Island_3_th_pred= get_prob(GMM_3_attribute.means, GMM_3.num_components, GMM_3_attribute.means, GMM_3_attribute.covariances, GMM_3_attribute.component_probs, factor=std_threshold_dict['tau1']).squeeze()
    Island_4_th_pred= get_prob(GMM_4_attribute.means, GMM_4.num_components, GMM_4_attribute.means, GMM_4_attribute.covariances, GMM_4_attribute.component_probs, factor=std_threshold_dict['tau1']).squeeze()
    Island_5_th_pred= get_prob(GMM_5_attribute.means, GMM_5.num_components, GMM_5_attribute.means, GMM_5_attribute.covariances, GMM_5_attribute.component_probs, factor=std_threshold_dict['tau1']).squeeze()
    Island_6_th_pred= get_prob(GMM_6_attribute.means, GMM_6.num_components, GMM_6_attribute.means, GMM_6_attribute.covariances, GMM_6_attribute.component_probs, factor=std_threshold_dict['tau1']).squeeze()
    Island_7_th_pred= get_prob(GMM_7_attribute.means, GMM_7.num_components, GMM_7_attribute.means, GMM_7_attribute.covariances, GMM_7_attribute.component_probs, factor=std_threshold_dict['tau1']).squeeze()
    Island_8_th_pred= get_prob(GMM_8_attribute.means, GMM_8.num_components, GMM_8_attribute.means, GMM_8_attribute.covariances, GMM_8_attribute.component_probs, factor=std_threshold_dict['tau1']).squeeze()
    Island_9_th_pred= get_prob(GMM_9_attribute.means, GMM_9.num_components, GMM_9_attribute.means, GMM_9_attribute.covariances, GMM_9_attribute.component_probs, factor=std_threshold_dict['tau1']).squeeze()
    Island_10_th_pred= get_prob(GMM_10_attribute.means, GMM_10.num_components, GMM_10_attribute.means, GMM_10_attribute.covariances, GMM_10_attribute.component_probs, factor=std_threshold_dict['tau1']).squeeze()
    Island_11_th_pred= get_prob(GMM_11_attribute.means, GMM_11.num_components, GMM_11_attribute.means, GMM_11_attribute.covariances, GMM_11_attribute.component_probs, factor=std_threshold_dict['tau1']).squeeze()
    Island_12_th_pred= get_prob(GMM_12_attribute.means, GMM_12.num_components, GMM_12_attribute.means, GMM_12_attribute.covariances, GMM_12_attribute.component_probs, factor=std_threshold_dict['tau1']).squeeze()
    Island_13_th_pred= get_prob(GMM_13_attribute.means, GMM_13.num_components, GMM_13_attribute.means, GMM_13_attribute.covariances, GMM_13_attribute.component_probs, factor=std_threshold_dict['tau1']).squeeze()
    Island_14_th_pred= get_prob(GMM_14_attribute.means, GMM_14.num_components, GMM_14_attribute.means, GMM_14_attribute.covariances, GMM_14_attribute.component_probs, factor=std_threshold_dict['tau1']).squeeze()
    Island_15_th_pred= get_prob(GMM_15_attribute.means, GMM_15.num_components, GMM_15_attribute.means, GMM_15_attribute.covariances, GMM_15_attribute.component_probs, factor=std_threshold_dict['tau1']).squeeze()
    Island_16_th_pred= get_prob(GMM_16_attribute.means, GMM_16.num_components, GMM_16_attribute.means, GMM_16_attribute.covariances, GMM_16_attribute.component_probs, factor=std_threshold_dict['tau1']).squeeze()
    Island_17_th_pred= get_prob(GMM_17_attribute.means, GMM_17.num_components, GMM_17_attribute.means, GMM_17_attribute.covariances, GMM_17_attribute.component_probs, factor=std_threshold_dict['tau1']).squeeze()
    Island_18_th_pred= get_prob(GMM_18_attribute.means, GMM_18.num_components, GMM_18_attribute.means, GMM_18_attribute.covariances, GMM_18_attribute.component_probs, factor=std_threshold_dict['tau1']).squeeze()
    Island_19_th_pred= get_prob(GMM_19_attribute.means, GMM_19.num_components, GMM_19_attribute.means, GMM_19_attribute.covariances, GMM_19_attribute.component_probs, factor=std_threshold_dict['tau1']).squeeze()
    Island_20_th_pred= get_prob(GMM_20_attribute.means, GMM_20.num_components, GMM_20_attribute.means, GMM_20_attribute.covariances, GMM_20_attribute.component_probs, factor=std_threshold_dict['tau1']).squeeze()
    Island_21_th_pred= get_prob(GMM_21_attribute.means, GMM_21.num_components, GMM_21_attribute.means, GMM_21_attribute.covariances, GMM_21_attribute.component_probs,factor=std_threshold_dict['tau1']).squeeze()
    Island_22_th_pred= get_prob(GMM_22_attribute.means, GMM_22.num_components, GMM_22_attribute.means, GMM_22_attribute.covariances, GMM_22_attribute.component_probs,factor=std_threshold_dict['tau1']).squeeze()
    
    print('compute probability P2')
    Island_0_th_pred3= get_prob(GMM_0_attribute.means, GMM_0.num_components, GMM_0_attribute.means, GMM_0_attribute.covariances, GMM_0_attribute.component_probs, factor=std_threshold_dict['tau2']).squeeze()
    Island_1_th_pred3= get_prob(GMM_1_attribute.means, GMM_1.num_components, GMM_1_attribute.means, GMM_1_attribute.covariances, GMM_1_attribute.component_probs, factor=std_threshold_dict['tau2']).squeeze()
    Island_2_th_pred3= get_prob(GMM_2_attribute.means, GMM_2.num_components, GMM_2_attribute.means, GMM_2_attribute.covariances, GMM_2_attribute.component_probs, factor=std_threshold_dict['tau2']).squeeze()
    Island_3_th_pred3= get_prob(GMM_3_attribute.means, GMM_3.num_components, GMM_3_attribute.means, GMM_3_attribute.covariances, GMM_3_attribute.component_probs, factor=std_threshold_dict['tau2']).squeeze()
    Island_4_th_pred3= get_prob(GMM_4_attribute.means, GMM_4.num_components, GMM_4_attribute.means, GMM_4_attribute.covariances, GMM_4_attribute.component_probs, factor=std_threshold_dict['tau2']).squeeze()
    Island_5_th_pred3= get_prob(GMM_5_attribute.means, GMM_5.num_components, GMM_5_attribute.means, GMM_5_attribute.covariances, GMM_5_attribute.component_probs, factor=std_threshold_dict['tau2']).squeeze()
    Island_6_th_pred3= get_prob(GMM_6_attribute.means, GMM_6.num_components, GMM_6_attribute.means, GMM_6_attribute.covariances, GMM_6_attribute.component_probs, factor=std_threshold_dict['tau2']).squeeze()
    Island_7_th_pred3= get_prob(GMM_7_attribute.means, GMM_7.num_components, GMM_7_attribute.means, GMM_7_attribute.covariances, GMM_7_attribute.component_probs, factor=std_threshold_dict['tau2']).squeeze()
    Island_8_th_pred3= get_prob(GMM_8_attribute.means, GMM_8.num_components, GMM_8_attribute.means, GMM_8_attribute.covariances, GMM_8_attribute.component_probs, factor=std_threshold_dict['tau2']).squeeze()
    Island_9_th_pred3= get_prob(GMM_9_attribute.means, GMM_9.num_components, GMM_9_attribute.means, GMM_9_attribute.covariances, GMM_9_attribute.component_probs, factor=std_threshold_dict['tau2']).squeeze()
    Island_10_th_pred3= get_prob(GMM_10_attribute.means, GMM_10.num_components, GMM_10_attribute.means, GMM_10_attribute.covariances, GMM_10_attribute.component_probs, factor=std_threshold_dict['tau2']).squeeze()
    Island_11_th_pred3= get_prob(GMM_11_attribute.means, GMM_11.num_components, GMM_11_attribute.means, GMM_11_attribute.covariances, GMM_11_attribute.component_probs, factor=std_threshold_dict['tau2']).squeeze()
    Island_12_th_pred3= get_prob(GMM_12_attribute.means, GMM_12.num_components, GMM_12_attribute.means, GMM_12_attribute.covariances, GMM_12_attribute.component_probs, factor=std_threshold_dict['tau2']).squeeze()
    Island_13_th_pred3= get_prob(GMM_13_attribute.means, GMM_13.num_components, GMM_13_attribute.means, GMM_13_attribute.covariances, GMM_13_attribute.component_probs, factor=std_threshold_dict['tau2']).squeeze()
    Island_14_th_pred3= get_prob(GMM_14_attribute.means, GMM_14.num_components, GMM_14_attribute.means, GMM_14_attribute.covariances, GMM_14_attribute.component_probs, factor=std_threshold_dict['tau2']).squeeze()
    Island_15_th_pred3= get_prob(GMM_15_attribute.means, GMM_15.num_components, GMM_15_attribute.means, GMM_15_attribute.covariances, GMM_15_attribute.component_probs, factor=std_threshold_dict['tau2']).squeeze()
    Island_16_th_pred3= get_prob(GMM_16_attribute.means, GMM_16.num_components, GMM_16_attribute.means, GMM_16_attribute.covariances, GMM_16_attribute.component_probs, factor=std_threshold_dict['tau2']).squeeze()
    Island_17_th_pred3= get_prob(GMM_17_attribute.means, GMM_17.num_components, GMM_17_attribute.means, GMM_17_attribute.covariances, GMM_17_attribute.component_probs, factor=std_threshold_dict['tau2']).squeeze()
    Island_18_th_pred3= get_prob(GMM_18_attribute.means, GMM_18.num_components, GMM_18_attribute.means, GMM_18_attribute.covariances, GMM_18_attribute.component_probs, factor=std_threshold_dict['tau2']).squeeze()
    Island_19_th_pred3= get_prob(GMM_19_attribute.means, GMM_19.num_components, GMM_19_attribute.means, GMM_19_attribute.covariances, GMM_19_attribute.component_probs, factor=std_threshold_dict['tau2']).squeeze()
    Island_20_th_pred3= get_prob(GMM_20_attribute.means, GMM_20.num_components, GMM_20_attribute.means, GMM_20_attribute.covariances, GMM_20_attribute.component_probs, factor=std_threshold_dict['tau2']).squeeze()
    Island_21_th_pred3= get_prob(GMM_21_attribute.means, GMM_21.num_components, GMM_21_attribute.means, GMM_21_attribute.covariances, GMM_21_attribute.component_probs,factor=std_threshold_dict['tau2']).squeeze()
    Island_22_th_pred3= get_prob(GMM_22_attribute.means, GMM_22.num_components, GMM_22_attribute.means, GMM_22_attribute.covariances, GMM_22_attribute.component_probs,factor=std_threshold_dict['tau2']).squeeze()
    print('get P2 and P3')
    Island_0_prob1, Island_0_prob2 = pred(opt, 0,Island_0_th_pred ,Island_0_th_pred3)
    Island_1_prob1, Island_1_prob2 = pred(opt, 1,Island_1_th_pred ,Island_1_th_pred3)
    Island_2_prob1, Island_2_prob2 = pred(opt, 2,Island_2_th_pred ,Island_2_th_pred3)
    Island_3_prob1, Island_3_prob2 = pred(opt, 3,Island_3_th_pred ,Island_3_th_pred3)
    Island_4_prob1, Island_4_prob2 = pred(opt, 4,Island_4_th_pred ,Island_4_th_pred3)
    Island_5_prob1, Island_5_prob2 = pred(opt, 5,Island_5_th_pred ,Island_5_th_pred3)
    Island_6_prob1, Island_6_prob2 = pred(opt, 6,Island_6_th_pred ,Island_6_th_pred3)
    Island_7_prob1, Island_7_prob2 = pred(opt, 7,Island_7_th_pred ,Island_7_th_pred3)
    Island_8_prob1, Island_8_prob2 = pred(opt, 8,Island_8_th_pred ,Island_8_th_pred3)
    Island_9_prob1, Island_9_prob2 = pred(opt, 9,Island_9_th_pred ,Island_9_th_pred3)
    Island_10_prob1, Island_10_prob2 = pred(opt, 10,Island_10_th_pred ,Island_10_th_pred3)
    Island_11_prob1, Island_11_prob2 = pred(opt, 11,Island_11_th_pred ,Island_11_th_pred3)
    Island_12_prob1, Island_12_prob2 = pred(opt, 12,Island_12_th_pred ,Island_12_th_pred3)
    Island_13_prob1, Island_13_prob2 = pred(opt, 13,Island_13_th_pred ,Island_13_th_pred3)
    Island_14_prob1, Island_14_prob2 = pred(opt, 14,Island_14_th_pred ,Island_14_th_pred3)
    Island_15_prob1, Island_15_prob2 = pred(opt, 15,Island_15_th_pred ,Island_15_th_pred3)
    Island_16_prob1, Island_16_prob2 = pred(opt, 16,Island_16_th_pred ,Island_16_th_pred3)
    Island_17_prob1, Island_17_prob2 = pred(opt, 17,Island_17_th_pred ,Island_17_th_pred3)
    Island_18_prob1, Island_18_prob2 = pred(opt, 18,Island_18_th_pred ,Island_18_th_pred3)
    Island_19_prob1, Island_19_prob2 = pred(opt, 19,Island_19_th_pred ,Island_19_th_pred3)
    Island_20_prob1, Island_20_prob2 = pred(opt, 20,Island_20_th_pred ,Island_20_th_pred3)
    Island_21_prob1, Island_21_prob2 = pred(opt, 21,Island_21_th_pred ,Island_21_th_pred3)
    Island_22_prob1, Island_22_prob2 = pred(opt, 22,Island_22_th_pred ,Island_22_th_pred3)
    
def main():
    opt = parse_option()    
    
    model, criterion = set_model(opt)
    GMM_0_attribute,GMM_1_attribute,GMM_2_attribute,GMM_3_attribute,GMM_4_attribute,GMM_5_attribute,GMM_6_attribute,GMM_7_attribute,GMM_8_attribute,GMM_9_attribute,GMM_10_attribute,GMM_11_attribute,GMM_12_attribute,GMM_13_attribute,GMM_14_attribute,GMM_15_attribute,GMM_16_attribute,GMM_17_attribute,GMM_18_attribute,GMM_19_attribute,GMM_20_attribute,GMM_21_attribute,GMM_22_attribute, GMM_0, GMM_1, GMM_2, GMM_3, GMM_4, GMM_5, GMM_6, GMM_7, GMM_8, GMM_9, GMM_10, GMM_11, GMM_12, GMM_13, GMM_14, GMM_15, GMM_16, GMM_17, GMM_18, GMM_19, GMM_20, GMM_21, GMM_22 = load_GMM_model(opt)
    
    GMM_Predict_prob(opt, GMM_0_attribute,GMM_1_attribute,GMM_2_attribute,GMM_3_attribute,GMM_4_attribute,GMM_5_attribute,GMM_6_attribute,GMM_7_attribute,GMM_8_attribute,GMM_9_attribute,GMM_10_attribute,GMM_11_attribute,GMM_12_attribute,GMM_13_attribute,GMM_14_attribute,GMM_15_attribute,GMM_16_attribute,GMM_17_attribute,GMM_18_attribute,GMM_19_attribute,GMM_20_attribute,GMM_21_attribute,GMM_22_attribute, GMM_0, GMM_1, GMM_2, GMM_3, GMM_4, GMM_5, GMM_6, GMM_7, GMM_8, GMM_9, GMM_10, GMM_11, GMM_12, GMM_13, GMM_14, GMM_15, GMM_16, GMM_17, GMM_18, GMM_19, GMM_20, GMM_21, GMM_22)

    
    
if __name__ == '__main__':
    main()
