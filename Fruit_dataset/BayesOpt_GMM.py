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
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt import SequentialDomainReductionTransformer

tau = (0, 1)
tau1 = (0, 1)
tau2 = (0, 1)

pbounds = {'tau0': tau, 'tau1': tau1, 'tau2': tau2}


if not os.path.isdir('./output/train_val/'):
    os.makedirs('./output/train_val/')

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
    parser.add_argument('--dataset', type=str, default='fruit_8',
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
    parser.add_argument('--gaussian_num', type=int, default=5,
                        help="gaussian number")
    
    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models_classifier'.format(opt.dataset)

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'fruit_8':
        opt.n_cls = 2
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    return opt

def set_loader(opt):
    # construct data loader
    if opt.dataset == 'fruit_8':
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
    if opt.dataset == 'fruit_8':
        _ ,_ ,_, _, _, _, _, train_df, train_com_df, val_df, test_df= get_fruit_8(seed=opt.seed)
        train_component_label = val_component_label = test_component_label = [0,1,2,3,4,5,6,7]
        
        val_df = pd.concat([val_df, train_df])        
        test_df_mapping2_label = val_df.copy()    
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
#         if torch.cuda.device_count() > 1:
#             model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict_model)

    return model, criterion

def calculatePerformance(df, file_name):
    df['overkill_rate'] = (df['overkill'] / df['total']).round(decimals = 5).astype(str)
    df['leakage_rate'] = (df['leakage'] / df['total']).round(decimals = 5).astype(str)
    df['unknown_rate'] = (df['unknown'] / df['total']).round(decimals = 5).astype(str)
    df = pd.concat([df, pd.DataFrame.from_records([
            {'total':sum(df['total']),
             'good':sum(df['good']),
             'bad':sum(df['bad']),
             'overkill':sum(df['overkill']), 
             'leakage':sum(df['leakage']), 
             'unknown':sum(df['unknown']), 
             'overkill_rate':round(1*(sum(df['overkill'])/sum(df['total'])),5), 
             'leakage_rate': round(1*(sum(df['leakage'])/sum(df['total'])),5), 
            'unknown_rate': round(1*(sum(df['unknown'])/sum(df['total'])),5)
            }
    ]
    )], sort=False)
    score =  1 / (3 / ( 1/np.exp(float(df['overkill_rate'].tail(1))) + 1/np.exp(float(df['leakage_rate'].tail(1))) + 1/np.exp(float(df['unknown_rate'].tail(1)))))
    df.to_csv(file_name, index=False)
    return score

def plotConfusionMatrix(y_true, y_pred, file_name):
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix)
    fx = sns.heatmap(df_cm, annot=True, fmt="d", cmap=plt.cm.Blues)
    try:
        fx.xaxis.set_ticklabels(['Good','Bad'])
        fx.yaxis.set_ticklabels(['Good','Bad'])    
    except:
        pass
    plt.figure(figsize = (9,6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap=plt.cm.Blues)
    plt.xlabel("prediction")
    plt.ylabel("label (ground truth)")
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()    

def plotConfusionMatrixUnknown(y_true, y_pred, file_name):
    y_true = pd.Series(y_true, name='True')  
    y_pred = pd.Series(y_pred, name='Predicted')
    df_cm = pd.crosstab(y_true, y_pred)
    plt.figure(figsize = (9,6))
    fx = sns.heatmap(df_cm, annot=True, fmt="d", cmap=plt.cm.Blues)
    try:
        fx.xaxis.set_ticklabels(['Good','Bad', 'Unknown'])
        fx.yaxis.set_ticklabels(['Good','Bad'])
    except:
        try:
            fx.xaxis.set_ticklabels(['Good','Bad'])
            fx.yaxis.set_ticklabels(['Good','Bad'])
        except:
            pass
    plt.xlabel("prediction")
    plt.ylabel("label (ground truth)")
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

def load_GMM_model(opt):
    gmm=GaussianMixture()

    GMM_0_attribute  = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+_fruit_8/batch_size128_gmm{opt.gaussian_num}/dataset_{opt.seed}_GMM_0_Classifier", "model"))
    GMM_1_attribute  = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+_fruit_8/batch_size128_gmm{opt.gaussian_num}/dataset_{opt.seed}_GMM_1_Classifier", "model"))
    GMM_2_attribute  = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+_fruit_8/batch_size128_gmm{opt.gaussian_num}/dataset_{opt.seed}_GMM_2_Classifier", "model"))
    GMM_3_attribute  = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+_fruit_8/batch_size128_gmm{opt.gaussian_num}/dataset_{opt.seed}_GMM_3_Classifier", "model"))
    GMM_4_attribute  = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+_fruit_8/batch_size128_gmm{opt.gaussian_num}/dataset_{opt.seed}_GMM_4_Classifier", "model"))
    GMM_5_attribute  = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+_fruit_8/batch_size128_gmm{opt.gaussian_num}/dataset_{opt.seed}_GMM_5_Classifier", "model"))
    GMM_6_attribute  = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+_fruit_8/batch_size128_gmm{opt.gaussian_num}/dataset_{opt.seed}_GMM_6_Classifier", "model"))
    GMM_7_attribute  = GaussianMixtureModel.load(os.path.join(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+_fruit_8/batch_size128_gmm{opt.gaussian_num}/dataset_{opt.seed}_GMM_7_Classifier", "model"))

    
    GMM_0 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+_fruit_8/batch_size128_gmm{opt.gaussian_num}/dataset_{opt.seed}_GMM_0_Classifier")
    GMM_1 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+_fruit_8/batch_size128_gmm{opt.gaussian_num}/dataset_{opt.seed}_GMM_1_Classifier")
    GMM_2 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+_fruit_8/batch_size128_gmm{opt.gaussian_num}/dataset_{opt.seed}_GMM_2_Classifier")
    GMM_3 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+_fruit_8/batch_size128_gmm{opt.gaussian_num}/dataset_{opt.seed}_GMM_3_Classifier")
    GMM_4 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+_fruit_8/batch_size128_gmm{opt.gaussian_num}/dataset_{opt.seed}_GMM_4_Classifier")
    GMM_5 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+_fruit_8/batch_size128_gmm{opt.gaussian_num}/dataset_{opt.seed}_GMM_5_Classifier")
    GMM_6 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+_fruit_8/batch_size128_gmm{opt.gaussian_num}/dataset_{opt.seed}_GMM_6_Classifier")
    GMM_7 = gmm.load(f"/root/notebooks/nfs/work/yanwei.liu/GMM/HybridExpert+_fruit_8/batch_size128_gmm{opt.gaussian_num}/dataset_{opt.seed}_GMM_7_Classifier")
    

    return GMM_0_attribute,GMM_1_attribute,GMM_2_attribute,GMM_3_attribute,GMM_4_attribute,GMM_5_attribute,GMM_6_attribute,GMM_7_attribute, GMM_0,GMM_1,GMM_2,GMM_3,GMM_4,GMM_5,GMM_6,GMM_7
def get_prob(X, k, mu, cov, phi, factor=0.1):
    X = X.cuda()
    mu = mu.cuda()
    cov = cov.cuda()
    likelihood = torch.zeros(X.shape[0], k)
    factor = factor + 0.001
    for i in tqdm(range(k)):
        distribution = multivariate_normal.MultivariateNormal(loc=mu[i], covariance_matrix=cov[i]*factor*factor)
        likelihood[:,i] = distribution.log_prob(X)

    numerator = likelihood * phi                         
    denominator = numerator.sum(axis=1)[:, np.newaxis] 
    weights = numerator / (denominator)
    return torch.exp(weights)

def pred(p_X, probs, probs_3_std):    
    p_X = np.where(p_X.numpy() > 1, 0, p_X)
    probs = np.where(probs.numpy() > 1, 0, probs)
    probs_3_std = np.where(probs_3_std.numpy() > 1, 0, probs_3_std)
    
    std2 = []
    std3 = []
    pred_result = []
    
    for ii in range(len(probs)):
        std2.append(probs[ii][ii])

    for ii in range(len(probs_3_std)):
        std3.append(probs_3_std[ii][ii])

    for i in range(len(p_X)):
        pred_good = p_X[i] >= std2
        pred_bad = (p_X[i] < std2) & (p_X[i] >= std3)
        pred_unk = p_X[i] < std3

        for n in range(len(pred_good)):
            do_once = 0
            if pred_good[n] == True:
                do_once += 1
                pred_result.append(0)
                break
        if do_once ==1:
            continue  
        for n in range(len(pred_bad)):
            if pred_bad[n] == True:
                pred_result.append(1)
                break
        a_3 = []
        
        for ii in range(len(pred_unk)):
            a_3.append(pred_unk[ii].item())
            
        if set(a_3) == {True}:
            pred_result.append(2)
    return pred_result

def GMM_Predict_prob(opt, features, GMM_0_attribute, GMM_1_attribute, GMM_2_attribute, GMM_3_attribute, GMM_4_attribute, GMM_5_attribute, GMM_6_attribute, GMM_7_attribute,GMM_0,GMM_1,GMM_2,GMM_3,GMM_4,GMM_5,GMM_6,GMM_7,tau0,tau1,tau2):
    print('Compute features prob')    
    features = torch.from_numpy(features)     
    Island_0_pred= get_prob(features, GMM_0.num_components, GMM_0_attribute.means, GMM_0_attribute.covariances, GMM_0_attribute.component_probs, factor=tau0).squeeze()
    Island_1_pred= get_prob(features, GMM_1.num_components, GMM_1_attribute.means, GMM_1_attribute.covariances, GMM_1_attribute.component_probs, factor=tau0).squeeze()
    Island_2_pred= get_prob(features, GMM_2.num_components, GMM_2_attribute.means, GMM_2_attribute.covariances, GMM_2_attribute.component_probs, factor=tau0).squeeze()
    Island_3_pred= get_prob(features, GMM_3.num_components, GMM_3_attribute.means, GMM_3_attribute.covariances, GMM_3_attribute.component_probs, factor=tau0).squeeze()
    Island_4_pred= get_prob(features, GMM_4.num_components, GMM_4_attribute.means, GMM_4_attribute.covariances, GMM_4_attribute.component_probs, factor=tau0).squeeze()
    Island_5_pred= get_prob(features, GMM_5.num_components, GMM_5_attribute.means, GMM_5_attribute.covariances, GMM_5_attribute.component_probs, factor=tau0).squeeze()
    Island_6_pred= get_prob(features, GMM_6.num_components, GMM_6_attribute.means, GMM_6_attribute.covariances, GMM_6_attribute.component_probs, factor=tau0).squeeze()
    Island_7_pred= get_prob(features, GMM_7.num_components, GMM_7_attribute.means, GMM_7_attribute.covariances, GMM_7_attribute.component_probs, factor=tau0).squeeze()

    
    print('compute 1-std th prob')
    Island_0_th_pred= get_prob(GMM_0_attribute.means, GMM_0.num_components, GMM_0_attribute.means, GMM_0_attribute.covariances, GMM_0_attribute.component_probs,factor=tau1).squeeze()
    Island_1_th_pred= get_prob(GMM_1_attribute.means, GMM_1.num_components, GMM_1_attribute.means, GMM_1_attribute.covariances, GMM_1_attribute.component_probs,factor=tau1).squeeze()
    Island_2_th_pred= get_prob(GMM_2_attribute.means, GMM_2.num_components, GMM_2_attribute.means, GMM_2_attribute.covariances, GMM_2_attribute.component_probs,factor=tau1).squeeze()
    Island_3_th_pred= get_prob(GMM_3_attribute.means, GMM_3.num_components, GMM_3_attribute.means, GMM_3_attribute.covariances, GMM_3_attribute.component_probs,factor=tau1).squeeze()
    Island_4_th_pred= get_prob(GMM_4_attribute.means, GMM_4.num_components, GMM_4_attribute.means, GMM_4_attribute.covariances, GMM_4_attribute.component_probs,factor=tau1).squeeze()
    Island_5_th_pred= get_prob(GMM_5_attribute.means, GMM_5.num_components, GMM_5_attribute.means, GMM_5_attribute.covariances, GMM_5_attribute.component_probs,factor=tau1).squeeze()
    Island_6_th_pred= get_prob(GMM_6_attribute.means, GMM_6.num_components, GMM_6_attribute.means, GMM_6_attribute.covariances, GMM_6_attribute.component_probs,factor=tau1).squeeze()
    Island_7_th_pred= get_prob(GMM_7_attribute.means, GMM_7.num_components, GMM_7_attribute.means, GMM_7_attribute.covariances, GMM_7_attribute.component_probs,factor=tau1).squeeze()

    
    print('compute 3-std th prob')
    Island_0_th_pred3= get_prob(GMM_0_attribute.means, GMM_0.num_components, GMM_0_attribute.means, GMM_0_attribute.covariances, GMM_0_attribute.component_probs,factor=tau2).squeeze()
    Island_1_th_pred3= get_prob(GMM_1_attribute.means, GMM_1.num_components, GMM_1_attribute.means, GMM_1_attribute.covariances, GMM_1_attribute.component_probs,factor=tau2).squeeze()
    Island_2_th_pred3= get_prob(GMM_2_attribute.means, GMM_2.num_components, GMM_2_attribute.means, GMM_2_attribute.covariances, GMM_2_attribute.component_probs,factor=tau2).squeeze()
    Island_3_th_pred3= get_prob(GMM_3_attribute.means, GMM_3.num_components, GMM_3_attribute.means, GMM_3_attribute.covariances, GMM_3_attribute.component_probs,factor=tau2).squeeze()
    Island_4_th_pred3= get_prob(GMM_4_attribute.means, GMM_4.num_components, GMM_4_attribute.means, GMM_4_attribute.covariances, GMM_4_attribute.component_probs,factor=tau2).squeeze()
    Island_5_th_pred3= get_prob(GMM_5_attribute.means, GMM_5.num_components, GMM_5_attribute.means, GMM_5_attribute.covariances, GMM_5_attribute.component_probs,factor=tau2).squeeze()
    Island_6_th_pred3= get_prob(GMM_6_attribute.means, GMM_6.num_components, GMM_6_attribute.means, GMM_6_attribute.covariances, GMM_6_attribute.component_probs,factor=tau2).squeeze()
    Island_7_th_pred3= get_prob(GMM_7_attribute.means, GMM_7.num_components, GMM_7_attribute.means, GMM_7_attribute.covariances, GMM_7_attribute.component_probs,factor=tau2).squeeze()

    
    print('features Classification')
    Island_0_prob = pred(Island_0_pred ,Island_0_th_pred ,Island_0_th_pred3)
    Island_1_prob = pred(Island_1_pred ,Island_1_th_pred ,Island_1_th_pred3)
    Island_2_prob = pred(Island_2_pred ,Island_2_th_pred ,Island_2_th_pred3)
    Island_3_prob = pred(Island_3_pred ,Island_3_th_pred ,Island_3_th_pred3)
    Island_4_prob = pred(Island_4_pred ,Island_4_th_pred ,Island_4_th_pred3)
    Island_5_prob = pred(Island_5_pred ,Island_5_th_pred ,Island_5_th_pred3)
    Island_6_prob = pred(Island_6_pred ,Island_6_th_pred ,Island_6_th_pred3)
    Island_7_prob = pred(Island_7_pred ,Island_7_th_pred ,Island_7_th_pred3)
  
        
    return Island_0_prob, Island_1_prob, Island_2_prob, Island_3_prob, Island_4_prob, Island_5_prob, Island_6_prob, Island_7_prob 


def GMM_prediction(label, Island_0_prob, Island_1_prob, Island_2_prob, Island_3_prob, Island_4_prob, Island_5_prob, Island_6_prob, Island_7_prob ,train_component_label, val_component_label, test_component_label,name_list):


    
    y_true = []
    y_pred = []
    
    c_name = []
    GMM_prediction_after_unknown = [] # final good or bad prediction to compare with class classifier [good, bad]
    gt_label_after_unknown = [] # [good, bad]
    
    # Good Island
    for idx, (result0, result1, result2, result3, result4, result5, result6, result7, gt, name) in enumerate(zip(Island_0_prob, Island_1_prob, Island_2_prob, Island_3_prob, Island_4_prob, Island_5_prob, Island_6_prob, Island_7_prob, label,name_list)):
        
        prediction_list = []
        prediction_list.extend([result0, result1, result2, result3, result4, result5, result6, result7])
        prediction_list_np = np.asarray(prediction_list)
        
        c_name.append(name)
        if 0 in set(prediction_list_np):
            y_true.append(gt)
            gt_label_after_unknown.append(gt)
            y_pred.append(0)
            GMM_prediction_after_unknown.append(0)
        else:
            if 1 in set(prediction_list_np):
                y_true.append(gt)
                gt_label_after_unknown.append(gt)
                y_pred.append(1)
                GMM_prediction_after_unknown.append(1)
            else:
                y_true.append(gt)
                y_pred.append(2)
    
    return y_true, y_pred, GMM_prediction_after_unknown, gt_label_after_unknown, c_name


def Expert1Testing(val_loader, model, criterion, opt, df, train_component_label, val_component_label, test_component_label):
    """
    Use Expert 1 for classification
    """
    model.eval()
    
    df_all = df.copy()
    df_ind = df.copy()
    df_ood = df.copy()
    
    # 針對IND和OOD樣本，將dataframe中不屬於IND或OOD類別的row刪除，避免重複計算的錯誤
    for name in df_ind["component_name"].value_counts().index:
        if name not in train_component_label:
            df_ind = df_ind[df_ind["component_name"] != name]
    for name in df_ood["component_name"].value_counts().index:
        if name not in test_component_label:
            df_ood = df_ood[df_ood["component_name"] != name]

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    y_pred = []
    y_true = []
    y_true_IND = []
    y_pred_IND = []
    y_true_OOD = []
    y_pred_OOD = []
    with torch.no_grad():
        end = time.time()
        for idx, (images, labels, path, component_name) in enumerate(tqdm(val_loader)):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output, _ = model(images)
            _, prediction=torch.max(output.data, 1)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1 = accuracy(output, labels, topk=(1))
            top1.update(acc1[0].item(), bsz)
            
            y_pred.extend(prediction.view(-1).detach().cpu().numpy())            
            y_true.extend(labels.view(-1).detach().cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # ALL test samples
            for idx, (gt, pred, name) in enumerate(list(zip(labels.data, prediction.data, component_name))):
                if gt.item() == 0 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'overkill'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'overkill'] +=1
                if gt.item() == 1 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'leakage'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'leakage'] +=1

            # IND test samples
            for idx, (gt, pred, name) in enumerate(list(zip(labels.data, prediction.data, component_name))):
                if name in train_component_label:
                    y_true_IND.append(gt.item())
                    y_pred_IND.append(pred.item())

                    if gt.item() == 0 and gt.item() != pred.item():
                        if isinstance(name, str):
                            df_ind.loc[(df_ind["component_name"] == name), 'overkill'] +=1
                        else:
                            df_ind.loc[(df_ind["component_name"] == name.item()), 'overkill'] +=1
                    if gt.item() == 1 and gt.item() != pred.item():
                        if isinstance(name, str):
                            df_ind.loc[(df_ind["component_name"] == name), 'leakage'] +=1
                        else:
                            df_ind.loc[(df_ind["component_name"] == name.item()), 'leakage'] +=1            

             # OOD test samples
            for idx, (gt, pred, name) in enumerate(list(zip(labels.data, prediction.data, component_name))):
                if name in test_component_label:
                    y_true_OOD.append(gt.item())
                    y_pred_OOD.append(pred.item())

                    if gt.item() == 0 and gt.item() != pred.item():
                        if isinstance(name, str):
                            df_ood.loc[(df_ood["component_name"] == name), 'overkill'] +=1
                        else:
                            df_ood.loc[(df_ood["component_name"] == name.item()), 'overkill'] +=1
                    if gt.item() == 1 and gt.item() != pred.item():
                        if isinstance(name, str):
                            df_ood.loc[(df_ood["component_name"] == name), 'leakage'] +=1
                        else:
                            df_ood.loc[(df_ood["component_name"] == name.item()), 'leakage'] +=1            
            
    print("Expert1 Accuracy: {}\n".format(100*round(accuracy_score(y_true, y_pred),4)))        
    
    return y_true, y_pred

def GMM(features, label, name_list, opt, df, train_component_label, val_component_label, test_component_label, tau0, tau1, tau2, GMM_0_attribute,GMM_1_attribute,GMM_2_attribute,GMM_3_attribute,GMM_4_attribute,GMM_5_attribute,GMM_6_attribute,GMM_7_attribute, GMM_0, GMM_1, GMM_2, GMM_3, GMM_4, GMM_5, GMM_6, GMM_7):

    y_true_all = []
    y_pred_all = []
    y_true_IND = []
    y_pred_IND = []
    y_true_OOD = []
    y_pred_OOD = []
    
    
    df_all = df.copy()
    df_ind = df.copy()
    df_ood = df.copy()
    
    for name in df_ind["component_name"].value_counts().index:
        if name not in train_component_label:
            df_ind = df_ind[df_ind["component_name"] != name]
    for name in df_ood["component_name"].value_counts().index:
        if name not in test_component_label:
            df_ood = df_ood[df_ood["component_name"] != name]    
                
    Island_0_prob, Island_1_prob, Island_2_prob, Island_3_prob, Island_4_prob, Island_5_prob, Island_6_prob, Island_7_prob = GMM_Predict_prob(opt, features, GMM_0_attribute,GMM_1_attribute,GMM_2_attribute,GMM_3_attribute,GMM_4_attribute,GMM_5_attribute,GMM_6_attribute,GMM_7_attribute, GMM_0, GMM_1, GMM_2, GMM_3, GMM_4, GMM_5, GMM_6, GMM_7, tau0,tau1,tau2)
    
    y_true, prediction_Expert2, GMM_prediction_after_unknown, gt_label_after_unknown, c_name = GMM_prediction(label, Island_0_prob, Island_1_prob, Island_2_prob, Island_3_prob, Island_4_prob, Island_5_prob, Island_6_prob, Island_7_prob ,train_component_label, val_component_label, test_component_label,name_list)
    
    for idx, (pred, gt, name) in enumerate(zip(prediction_Expert2, y_true, c_name)): 
        # ALL test samples
        if pred == 2:
            y_true_all.append(gt)
            y_pred_all.append(2)
            if isinstance(name, int):
                df_all.loc[(df_all["component_name"] == name), 'unknown'] +=1
            else:
                df_all.loc[(df_all["component_name"] == ''.join(name)), 'unknown'] +=1
        if pred !=2:
            y_true_all.append(gt)
            y_pred_all.append(pred)
            if gt == 0 and gt != pred:
                if isinstance(name, int):
                    df_all.loc[(df_all["component_name"] == name), 'overkill'] +=1
                else:
                    df_all.loc[(df_all["component_name"] == name.item()), 'overkill'] +=1
            if gt == 1 and gt != pred:
                if isinstance(name, int):
                    df_all.loc[(df_all["component_name"] == name), 'leakage'] +=1
                else:
                    df_all.loc[(df_all["component_name"] == name.item()), 'leakage'] +=1
        
        # IND test samples
        if name in train_component_label:
            
            if pred == 2:
                y_true_IND.append(gt)
                y_pred_IND.append(2)
                if isinstance(name, int):
                    df_ind.loc[(df_ind["component_name"] == name), 'unknown'] +=1
                else:
                    df_ind.loc[(df_ind["component_name"] == ''.join(name)), 'unknown'] +=1
            
            if pred !=2:
                y_true_IND.append(gt)
                y_pred_IND.append(pred)
                if gt == 0 and gt != pred:    
                    if isinstance(name, int):
                        df_ind.loc[(df_ind["component_name"] == name), 'overkill'] +=1
                    else:
                        df_ind.loc[(df_ind["component_name"] == name.item()), 'overkill'] +=1
                if gt == 1 and gt != pred:
                    if isinstance(name, int):
                        df_ind.loc[(df_ind["component_name"] == name), 'leakage'] +=1
                    else:
                        df_ind.loc[(df_ind["component_name"] == name.item()), 'leakage'] +=1    
        
        # OOD test samples
        if name in test_component_label:
            if pred == 2:
                y_true_OOD.append(gt)
                y_pred_OOD.append(2)
                if isinstance(name, int):
                    df_ood.loc[(df_ood["component_name"] == name), 'unknown'] +=1
                else:
                    df_ood.loc[(df_ood["component_name"] == ''.join(name)), 'unknown'] +=1
            if pred !=2:                
                y_true_OOD.append(gt)
                y_pred_OOD.append(pred)
                if gt == 0 and gt != pred:
                    if isinstance(name, int):
                        df_ood.loc[(df_ood["component_name"] == name), 'overkill'] +=1
                    else:
                        df_ood.loc[(df_ood["component_name"] == name.item()), 'overkill'] +=1
                if gt == 1 and gt != pred:
                    if isinstance(name, int):
                        df_ood.loc[(df_ood["component_name"] == name), 'leakage'] +=1
                    else:
                        df_ood.loc[(df_ood["component_name"] == name.item()), 'leakage'] +=1    
    
    print("Expert2 accuracy: {}\n".format(100*round(accuracy_score(gt_label_after_unknown, GMM_prediction_after_unknown),4)))        
    
    return y_pred_all, gt_label_after_unknown, GMM_prediction_after_unknown

def HybridExpert(y_pred_expert1, y_pred_expert2, y_true, name_label_list, df, opt, train_component_label, val_component_label, test_component_label):
    
    
    gt_label_after_unknown = []
    pred_after_unknown = []
    
    y_pred_all = []
    y_true_all = []
    
    prediction_IND = []
    gt_label_IND = []
    prediction_OOD = []
    gt_label_OOD = []
    
    df_ind = df.copy()
    df_ood = df.copy()
    
    for name in df_ind["component_name"].value_counts().index:
        if name not in train_component_label:
            df_ind = df_ind[df_ind["component_name"] != name]
    for name in df_ood["component_name"].value_counts().index:
        if name not in test_component_label:
            df_ood = df_ood[df_ood["component_name"] != name]
            
    for idx, (pred_expert1, pred_expert2, gt_label, name) in enumerate(list(zip(y_pred_expert1, y_pred_expert2, y_true, name_label_list))):
        
        if (pred_expert1 != pred_expert2) or (pred_expert2 ==2):

            y_pred_all.append(2)
            y_true_all.append(gt_label)

            if isinstance(name, int):
                df.loc[(df["component_name"] == name), 'unknown'] +=1
            else:
                df.loc[(df["component_name"] == ''.join(name)), 'unknown'] +=1

        if pred_expert1 == pred_expert2:

            y_pred_all.append(pred_expert2)
            y_true_all.append(gt_label)
            
            gt_label_after_unknown.append(gt_label)
            pred_after_unknown.append(pred_expert2)

            if gt_label == 0 and gt_label != pred_expert2:
                if isinstance(name, int):
                    df.loc[(df["component_name"] == name), 'overkill'] +=1
                else:
                    df.loc[(df["component_name"] == ''.join(name)), 'overkill'] +=1
            elif gt_label == 1 and gt_label != pred_expert2:
                if isinstance(name, int):
                    df.loc[(df["component_name"] == name), 'leakage'] +=1
                else:
                    df.loc[(df["component_name"] == ''.join(name)), 'leakage'] +=1 
        
        if name in train_component_label:     # old component samples    
            if (pred_expert1 != pred_expert2) or (pred_expert2 ==2):
                
                prediction_IND.append(2)
                gt_label_IND.append(gt_label)
                
                if isinstance(name, int):
                    df_ind.loc[(df_ind["component_name"] == name), 'unknown'] +=1
                else:
                    df_ind.loc[(df_ind["component_name"] == ''.join(name)), 'unknown'] +=1
    
            if pred_expert1 == pred_expert2:
            
                prediction_IND.append(pred_expert2)
                gt_label_IND.append(gt_label)
        
                if gt_label == 0 and gt_label != pred_expert2:
                    if isinstance(name, int):
                        df_ind.loc[(df_ind["component_name"] == name), 'overkill'] +=1
                    else:
                        df_ind.loc[(df_ind["component_name"] == ''.join(name)), 'overkill'] +=1
                elif gt_label == 1 and gt_label != pred_expert2:
                    if isinstance(name, int):
                        df_ind.loc[(df_ind["component_name"] == name), 'leakage'] +=1
                    else:
                        df_ind.loc[(df_ind["component_name"] == ''.join(name)), 'leakage'] +=1         
        
        if name in test_component_label:     # new component samples    
            if (pred_expert1 != pred_expert2) or (pred_expert2 ==2):
                
                prediction_OOD.append(2)
                gt_label_OOD.append(gt_label)
                
                if isinstance(name, int):
                    df_ood.loc[(df_ood["component_name"] == name), 'unknown'] +=1
                else:
                    df_ood.loc[(df_ood["component_name"] == ''.join(name)), 'unknown'] +=1
    
            if pred_expert1 == pred_expert2:
            
                prediction_OOD.append(pred_expert2)
                gt_label_OOD.append(gt_label)
        
                if gt_label == 0 and gt_label != pred_expert2:
                    if isinstance(name, int):
                        df_ood.loc[(df_ood["component_name"] == name), 'overkill'] +=1
                    else:
                        df_ood.loc[(df_ood["component_name"] == ''.join(name)), 'overkill'] +=1
                elif gt_label == 1 and gt_label != pred_expert2:
                    if isinstance(name, int):
                        df_ood.loc[(df_ood["component_name"] == name), 'leakage'] +=1
                    else:
                        df_ood.loc[(df_ood["component_name"] == ''.join(name)), 'leakage'] +=1 
    
    print("HybridExpert Accuracy: {}\n".format(100*round(accuracy_score(gt_label_after_unknown, pred_after_unknown),4)))        

    score = calculatePerformance(df, file_name=f'./output/train_val/dataset_{opt.seed}_HBE+_ALL_train_val.csv')
    return score


opt = parse_option()    
val_loader, _, val_component_name_df, train_component_label, val_component_label, _ = set_loader(opt)

val_df_orig = val_component_name_df.copy()
val_component_name_df_Expert2 = val_df_orig.copy()
val_component_name_df_HybridExpert = val_df_orig.copy()
# build model and criterion
model, criterion = set_model(opt)

# Extracted features for GMM testings
features, gt_labels, image_paths, name_label_list, _, _ = get_features_trained_weight(model, val_loader, opt.embedding_layer)

print("1. Testing with Discriminative Model (Expert 1)")  # Discriminative model
y_true, y_pred_expert1 = Expert1Testing(val_loader, model, criterion, opt, val_component_name_df, train_component_label, val_component_label, val_component_label)

GMM_0_attribute,GMM_1_attribute,GMM_2_attribute,GMM_3_attribute,GMM_4_attribute,GMM_5_attribute,GMM_6_attribute,GMM_7_attribute, GMM_0, GMM_1, GMM_2, GMM_3, GMM_4, GMM_5, GMM_6, GMM_7= load_GMM_model(opt)

def search(tau0, tau1, tau2):
    
    val_component_name_df_HybridExpert = val_df_orig.copy()
    
    params = {}    
    params['tau0'] = tau0
    params['tau1'] = tau1
    params['tau2'] = tau2

    print("2. Testing with Generative Model (Expert 2)")      # Generative model GMM Classifier
    y_pred_expert2, gt_label_after_unknown, GMM_prediction_after_unknown = GMM(features, gt_labels, name_label_list, opt, val_component_name_df_Expert2,train_component_label, val_component_label, val_component_label, tau0, tau1, tau2, GMM_0_attribute,GMM_1_attribute,GMM_2_attribute,GMM_3_attribute,GMM_4_attribute,GMM_5_attribute,GMM_6_attribute,GMM_7_attribute, GMM_0, GMM_1, GMM_2, GMM_3, GMM_4, GMM_5, GMM_6, GMM_7)

    print("3. Testing with Hybrid Expert")      # Generative model GMM Classifier
    score = HybridExpert(y_pred_expert1, y_pred_expert2, y_true, name_label_list, val_component_name_df_HybridExpert, opt, train_component_label, val_component_label, val_component_label)
    return score    

# Run Bayesian Optimization
start = time.time()
logger = JSONLogger(path=f"./{opt.seed}_tau1_tau2_logs.json")
bounds_transformer = SequentialDomainReductionTransformer(minimum_window=0.5)
bayes_optimizer = BayesianOptimization(search, random_state=42, pbounds=pbounds, bounds_transformer=bounds_transformer)
bayes_optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
bayes_optimizer.maximize(init_points=100, n_iter=100, acq="ei", xi=0)
print('It takes %s minutes' % ((time.time() - start)/60))
print(bayes_optimizer.max)
