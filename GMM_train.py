from __future__ import print_function

import pickle
import joblib
import sys
import argparse
import time
import math
import os
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from util import *
import matplotlib.pyplot as plt
from networks.mobilenetv3 import SupConMobileNetV3Large
from pycave.bayes.gmm import GaussianMixture
from pycave.bayes.gmm.model import GaussianMixtureModel
import numpy as np
from tqdm import tqdm

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed for datasets')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='10,15,20',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
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

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument("--embedding_layer", type=str, default="shared_embedding", help="Which embedding to visualization( encoder or shared_embedding)")
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to final model')
    parser.add_argument('--componentName', type=int, default=0,
                        help="component name")
    parser.add_argument('--gaussian_num', type=int, default=5,
                        help="gaussian number")

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

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
        train_df = CreateDataset_relabel_for_each_component(opt.seed, opt.componentName)
        
        if opt.componentName == 21 or opt.componentName == 22:
            train_df = train_df         
        else:
            train_df = train_df.loc[(train_df['class'] == 0)]
        
        train_dataset = CustomDataset(train_df, 
                                      transform=val_transform)
    else:
        raise ValueError(opt.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True, sampler=None)

    return train_loader


def set_model(opt):
    model = SupConMobileNetV3Large(name=opt.model)
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict_model = ckpt['model']
    
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict_model)

    return model

def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model
    model = set_model(opt)
    
    features, labels, image_paths, name_list, _, _ = get_features_trained_weight(model, train_loader, opt.embedding_layer)    
    for idx, num in enumerate(set(name_list)):
        if idx != num:
            item = np.where(np.asarray(name_list) == num)[0]
            for i in item:
                name_list[i] = idx
    
    PATH = f"/home/a3ilab01/Desktop/GMM/HybridExpert+/dataset_{opt.seed}_GMM_{opt.componentName}_Classifier/"
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    gmm=GaussianMixture(num_components=opt.gaussian_num, covariance_type='full', trainer_params=dict(gpus=1), covariance_regularization=1e-3)
    try:
        gmm.fit(features)
        gmm.save(PATH)
    except:
        with open("GMM_not_positive-definite.txt", 'a') as f:
            f.write(f'dataset seed: {opt.seed}, component name: {opt.componentName}\n')

if __name__ == '__main__':
    main()