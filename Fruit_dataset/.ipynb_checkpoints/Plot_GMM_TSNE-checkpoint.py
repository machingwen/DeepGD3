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
from matplotlib import pyplot as plt
from pycave.bayes.gmm import GaussianMixture
from pycave.bayes.gmm.model import GaussianMixtureModel
import torchvision.transforms as T
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
import matplotlib.patheffects as PathEffects
import seaborn as sns
from pprint import pformat
from typing import Dict, Any, List, Tuple
import random

plt.rcParams['figure.figsize'] = (16, 16)
plt.rcParams['figure.dpi'] = 150

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
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to final model')
    parser.add_argument("--embedding_layer", type=str, default="shared_embedding", help="Which embedding to visualization( encoder or shared_embedding)")
    parser.add_argument("--output_dir", type=str, default="output/", help="Directory to save output plots")
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

def set_random_seed(seed):
    """
    Set random seed for package random, numpy and pytorch
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def compute_plot_coordinates(image,x,y,image_centers_area_size,offset):

    image_height, image_width, _ = image.shape
    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset
    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center, compute the coordinates of the top left and bottom right corner
    xmin = center_x - int(image_width / 2)
    ymin = center_y - int(image_height / 2)
    xmax = xmin + image_width
    ymax = ymin + image_height

    return xmin, ymin, xmax, ymax    

def plot_scatter_all(args, X, y, X_test, y_test):    
    random.seed(12345)
    tx = X[:, 0]
    ty = X[:, 1]
    
    tx_test = X_test[:, 0]
    ty_test = X_test[:, 1]
    
    figure = plt.figure()
    ax = plt.subplot(aspect="equal")
    n_classes = len(set(y))
    
    for idx, num in enumerate(set(y)):
        if idx != num:
            item = np.where(np.asarray(y) == num)[0]
            for i in item:
                y[i] = idx
    
    colors_per_class = {}
    
    for i in set(y):
        colors_per_class[i] = [random.randrange(0, 255) for i in range(3)]
    
    for label in colors_per_class:
        indices = [i for i, l in enumerate(y) if l == label]
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        color = np.array([colors_per_class[label][::-1]], dtype=float) / 255        
        ax.scatter(current_tx, current_ty, lw=0, s=40, c=color, label=label)
    
    colors_per_class = {}
    colors_per_class[len(set(y))] = [0, 0, 255]
    colors_per_class[len(set(y))+1] = [0, 255, 0]
    for idx, label in enumerate(colors_per_class):
        indices = [i for i, l in enumerate(y_test) if l == label]
        current_tx = np.take(tx_test, indices)
        current_ty = np.take(ty_test, indices)
        color = np.array([colors_per_class[label][::-1]], dtype=float) / 255        
        if idx == 0:
            ax.scatter(current_tx, current_ty, marker="o", c=color, label=label, s=250, edgecolors="k", linewidths=1.5, alpha=0.8) # good new component
        else:
            ax.scatter(current_tx, current_ty, marker="X", c=color, label=label, s=250, edgecolors="k", linewidths=1.5, alpha=0.8) # bad new component
    
    ax.axis("tight")
    ax.axis("off")
    
    # Add the labels for each sample corresponding to the label
    for i in range(n_classes):
        # Position of each label at median of data points
        j = (i+1)*200
        if i ==0:
            x_text, y_text = np.median(X[0:j], axis=0)
        else:
            x_text, y_text = np.median(X[j-200:j], axis=0)
        text = ax.text(x_text, y_text, str(i), fontsize=20)
        text.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()
        ])
    
    ax.legend(loc='best')
    tsne_points_path= os.path.join(args.output_dir, f"{args.seed}/{args.embedding_layer}_train+test_stage2_component_HBE+_component_tsne.pdf")
    plt.savefig(tsne_points_path, dpi=150)

def plot_scatter(args, X, y, X_test, y_test):    
    tx = X[:, 0]
    ty = X[:, 1]
    tx_test = X_test[:, 0]
    ty_test = X_test[:, 1]
    
    figure = plt.figure()
    ax = plt.subplot(aspect="equal")
    
    colors_per_class = {}
    colors_per_class[0] = [0, 0, 255]
    colors_per_class[1] = [255, 0, 0]
    
    for label in colors_per_class:
        indices = [i for i, l in enumerate(y) if l == label]
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        color = np.array([colors_per_class[label][::-1]], dtype=float) / 255        
        ax.scatter(current_tx, current_ty, lw=0, s=40, c=color, label=label)
        
    colors_per_class = {}
    colors_per_class[len(set(y))] = [0, 0, 255]
    colors_per_class[len(set(y))+1] = [0, 255, 0]
    for idx, label in enumerate(colors_per_class):
        indices = [i for i, l in enumerate(y_test) if l == label]
        current_tx = np.take(tx_test, indices)
        current_ty = np.take(ty_test, indices)
        color = np.array([colors_per_class[label][::-1]], dtype=float) / 255        
        if idx == 0:
            ax.scatter(current_tx, current_ty, marker="o", c=color, label=label, s=250, edgecolors="k", linewidths=1.5, alpha=0.8) # good new component
        else:
            ax.scatter(current_tx, current_ty, marker="X", c=color, label=label, s=250, edgecolors="k", linewidths=1.5, alpha=0.8) # bad new component
    
    ax.axis("tight")
    ax.axis("off")
    ax.legend(loc='best')
    tsne_points_path= os.path.join(args.output_dir, f"{args.seed}/{args.embedding_layer}_train+test_stage2_component_HBE+_class_tsne.pdf")
    plt.savefig(tsne_points_path, dpi=150)

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
        _ ,_ ,_, _, _ ,_, _,_ , train_df, val_df , test_df = get_fruit_8(root="./data" , seed=opt.seed)
        train_component_label = val_component_label = test_component_label = [0,1,2,3,4,5,6,7]
#         if opt.relabel:
#             train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, _ = CreateDataset_relabel(opt.seed, testing=True)
        
#         val_df = pd.concat([val_df, train_df])        
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
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict_model = ckpt['model']
    
    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True
        model.load_state_dict(state_dict_model)

    return model

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

    return GMM_0_attribute,GMM_1_attribute,GMM_2_attribute,GMM_3_attribute,GMM_4_attribute,GMM_5_attribute,GMM_6_attribute,GMM_7_attribute,GMM_0,GMM_1,GMM_2,GMM_3,GMM_4,GMM_5,GMM_6,GMM_7

def main():
    opt = parse_option()
    set_random_seed(opt.seed)
    model = set_model(opt)
    
    sample_num = 200
    total_num = sample_num*8
    
    GMM_0_attribute,GMM_1_attribute,GMM_2_attribute,GMM_3_attribute,GMM_4_attribute,GMM_5_attribute,GMM_6_attribute,GMM_7_attribute,GMM_0,GMM_1,GMM_2,GMM_3,GMM_4,GMM_5,GMM_6,GMM_7 = load_GMM_model(opt)
    
    _, _, test_loader = CreateTSNEdataset_regroup_fruit_8(opt.seed)
    embeddings_test, labels_test, _, name_list_test, _, _ = get_features_trained_weight(model, test_loader, embedding_layer=opt.embedding_layer, tsne=True)   
    
    feat_0 = GMM_0.sample(sample_num).numpy()
    feat_1 = GMM_1.sample(sample_num).numpy()
    feat_2 = GMM_2.sample(sample_num).numpy()
    feat_3 = GMM_3.sample(sample_num).numpy()
    feat_4 = GMM_4.sample(sample_num).numpy()
    feat_5 = GMM_5.sample(sample_num).numpy()
    feat_6 = GMM_6.sample(sample_num).numpy()
    feat_7 = GMM_7.sample(sample_num).numpy()

    
    label_0 = [0] * feat_0.shape[0]
    label_1 = [1] * feat_1.shape[0]
    label_2 = [2] * feat_2.shape[0]
    label_3 = [3] * feat_3.shape[0]
    label_4 = [4] * feat_4.shape[0]
    label_5 = [5] * feat_5.shape[0]
    label_6 = [6] * feat_6.shape[0]
    label_7 = [7] * feat_7.shape[0]

    
    embeddings = np.concatenate((feat_0, feat_1, feat_2, feat_3, feat_4, feat_5, feat_6, feat_7, embeddings_test), axis=0)
    name_list_train = label_0+label_1+label_2+label_3+label_4+label_5+label_6+label_7
    
    labels_train = [0] * (total_num)

    
    tsne = TSNE(n_components=2, random_state=12345, perplexity=35, learning_rate=200, n_iter=2000, n_jobs=-1)
    X_transformed = tsne.fit_transform(embeddings)
    
    tsne_train = X_transformed[0:total_num]
    tsne_test = X_transformed[total_num::]
    
    component_labels_test_mapping = []
    for lb in labels_test:
        if lb == 0:
            component_labels_test_mapping.append(len(set(name_list_train)))
        if lb == 1:
            component_labels_test_mapping.append(len(set(name_list_train))+1)
    
    class_labels_test_mapping = []
    for lb in labels_test:
        if lb == 0:
            class_labels_test_mapping.append(len(set(labels_train)))
        if lb == 1:
            class_labels_test_mapping.append(len(set(labels_train))+1)
    
    plot_scatter_all(opt, tsne_train, name_list_train, tsne_test, component_labels_test_mapping)
    plot_scatter(opt, tsne_train, labels_train, tsne_test, class_labels_test_mapping)

if __name__ == '__main__':
    main()
