# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import AverageMeter, accuracy
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer_two, save_model, CreateDataset_relabel, CustomDataset, get_fruit_8
from networks.mobilenetv3 import SupConMobileNetV3Large
from pytorch_metric_learning import samplers
from losses import MultiSimilarityLoss
from tqdm import tqdm
import random
import numpy as np


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=30,
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
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='mobilenetv3_large')
    parser.add_argument('--dataset', type=str, default='fruit_8')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=224, help='parameter for Resize')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    # temperature
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None
    
    opt.warm=True
    opt.cosine=True
    if opt.warm:
        opt.warmup_from = 0.01
        opt.warm_epochs = 5
        if opt.cosine:
            print("Cosine learning rate scheduler")
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate
            
    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)
    opt.vis_path = './output/{}'.format(opt.seed)
    
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, str(opt.seed), 'train')
    opt.tb_val_folder = os.path.join(opt.tb_path, str(opt.seed), 'val')
    opt.vis_folder = opt.vis_path
    
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
    if not os.path.isdir(opt.tb_val_folder):
        os.makedirs(opt.tb_val_folder)
    if not os.path.isdir(opt.vis_folder):
        os.makedirs(opt.vis_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name, str(opt.seed))
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def set_loader(opt):
    if opt.dataset == 'fruit_8':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.Resize([opt.size, opt.size]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])    
    train_com_transform = transforms.Compose([
        transforms.Resize([opt.size, opt.size]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize([opt.size, opt.size]),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'fruit_8':
        input_size ,num_classes ,train_com_loader, train_cls_loader, val_loader, val_com_loader, test_loader, train_cls_df, train_com_df, val_df, test_df= get_fruit_8(opt.seed)
        
    else:
        raise ValueError(opt.dataset)
    
#     per_component_num = opt.batch_size // len(train_com_dataset.dataframe['component_name'].value_counts().index)
#     per_class_num = opt.batch_size // len(train_cls_dataset.dataframe['class'].value_counts().index)
    
#     train_com_sampler = samplers.MPerClassSampler(train_com_dataset.dataframe['component_name'], per_component_num, length_before_new_iter=len(train_com_dataset))
#     train_cls_sampler = samplers.MPerClassSampler(train_cls_dataset.dataframe['class'], per_class_num, length_before_new_iter=len(train_cls_dataset))
    
#     train_com_loader = torch.utils.data.DataLoader(
#         train_com_dataset, batch_size=opt.batch_size, shuffle=(train_com_sampler is None),
#         num_workers=opt.num_workers, pin_memory=True, sampler=train_com_sampler)
    
#     train_cls_loader = torch.utils.data.DataLoader(
#         train_cls_dataset, batch_size=opt.batch_size, shuffle=(train_cls_sampler is None),
#         num_workers=opt.num_workers, pin_memory=True, sampler=train_cls_sampler)
    
#     val_loader = torch.utils.data.DataLoader(
#         test_dataset, batch_size=opt.batch_size, shuffle=False,
#         num_workers=opt.num_workers, pin_memory=True)

#     val_com_loader = torch.utils.data.DataLoader(
#         test_com_dataset, batch_size=opt.batch_size, shuffle=False,
#         num_workers=opt.num_workers, pin_memory=True)

    return train_com_loader, train_cls_loader, val_loader, val_com_loader


def set_model(opt):
    model = SupConMobileNetV3Large(name=opt.model)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion_com = MultiSimilarityLoss().cuda()
        criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        
    return model, criterion_com, criterion_cls


def train(train_com_loader, train_cls_loader, model, criterion_com, criterion_cls, optimizer_cls, optimizer_component, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_class = AverageMeter()
    losses_component = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    
    component_dataloader_iterator = iter(train_com_loader)
    
    for idx, (data2) in enumerate(train_cls_loader):
        data_time.update(time.time() - end)       
        
        try:
            data1 = next(component_dataloader_iterator)
        except StopIteration:
            component_dataloader_iterator = iter(train_com_loader)
            data1 = next(component_dataloader_iterator)        
        
        # Dataset
        images_com, _, _, component_name = data1
        images_cls, labels, _, _ = data2
        
        if torch.cuda.is_available():
            images_com = images_com.cuda(non_blocking=True)
            images_cls = images_cls.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            component_name = component_name.cuda(non_blocking=True)
        bsz = component_name.shape[0]
        
        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_com_loader), optimizer_component)
        warmup_learning_rate(opt, epoch, idx, len(train_cls_loader), optimizer_cls)
                
        """
        Class classifier branch
        """
        class_out, _ = model(images_cls)  # 對兩張相同但採用不同資料擴增方式的圖片，取得各自的class feature
        # SGD
        loss_cls = criterion_cls(class_out, labels)
        optimizer_cls.zero_grad()
        loss_cls.backward(retain_graph=True)
        optimizer_cls.step()
        
        """
        Component classifier branch
        """
        _, component_out = model(images_com)  # 對兩張相同但採用不同資料擴增方式的圖片，取得各自的class feature        
        # SGD
        loss_component = criterion_com(component_out, component_name)
        optimizer_component.zero_grad() #set the grade to zero
        loss_component.backward() 
        optimizer_component.step()
                
        # update metric
        losses.update((loss_cls + loss_component).item(), bsz)
        acc1 = accuracy(class_out, labels, topk=(1))
        top1.update(acc1[0].item(), bsz)
        losses_class.update(loss_cls.item(), bsz)
        losses_component.update(loss_component.item(), bsz)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t\t'
                  'loss class {loss_class.val:.3f} ({loss_class.avg:.3f})\t\t'
                  'loss component {loss_component.val:.3f} ({loss_component.avg:.3f})\t\t'
                  'Class Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t\t'.format(
                   epoch, idx + 1, len(train_cls_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, loss_class=losses_class, loss_component=losses_component, top1=top1))
            sys.stdout.flush()

    return losses.avg, losses_class.avg, losses_component.avg, top1.avg

def validate(val_loader, val_com_loader, model, criterion_com, criterion_cls, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_component = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels, _, component_name) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            component_name = component_name.cuda(non_blocking=True)
            bsz = component_name.shape[0]

            # forward
            output, component_out = model(images)
            loss = criterion_cls(output, labels)
            # update metric
            losses.update(loss.item(), bsz)
            acc1 = accuracy(output, labels, topk=(1))
            top1.update(acc1[0].item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss class {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Class Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    
        for idx, (images, labels, _, component_name) in enumerate(val_com_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            component_name = component_name.cuda(non_blocking=True)
            bsz = component_name.shape[0]

            # forward
            output, component_out = model(images)
            loss_component = criterion_com(component_out, component_name)
            # update metric

            losses_component.update(loss_component.item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss component {loss_component.val:.4f} ({loss_component.avg:.4f}\t'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss_component=losses_component))
    
    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    
    return losses.avg, losses_component.avg, top1.avg

def main():
    opt = parse_option()

    # build data loader
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    train_com_loader, train_cls_loader, val_loader, val_com_loader = set_loader(opt)

    # build model and criterion
    model, criterion_com, criterion_cls = set_model(opt)

    # build optimizer
    optimizer_cls, optimizer_component = set_optimizer_two(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    logger_val = tb_logger.Logger(logdir=opt.tb_val_folder, flush_secs=2)

    best_loss = float("inf")
    the_last_loss = float("inf")
    patience = 2
    trigger_times = 0
    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer_component, epoch)
        adjust_learning_rate(opt, optimizer_cls, epoch)        

        # train for one epoch
        time1 = time.time()
        loss, cls_loss, component_loss, train_acc = train(train_com_loader, train_cls_loader, model, criterion_com, criterion_cls, optimizer_cls, optimizer_component, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}, train accuracy:{:.2f}'.format(epoch, time2 - time1, train_acc))
        
        # tensorboard logger
        logger.log_value('total loss', loss, epoch)
        logger.log_value('component loss', component_loss, epoch)
        logger.log_value('class loss', cls_loss, epoch)
        logger.log_value('class acc', train_acc, epoch)
        logger.log_value('learning_rate', optimizer_component.param_groups[0]['lr'], epoch)
        
        # eval for one epoch
        val_cls_loss, val_component_loss, val_acc = validate(val_loader, val_com_loader, model, criterion_com, criterion_cls, opt)
        
        # tensorboard logger
        val_loss = val_cls_loss + val_component_loss
        logger_val.log_value('total loss', val_loss, epoch)
        logger_val.log_value('component loss', val_component_loss, epoch)
        logger_val.log_value('class loss', val_cls_loss, epoch)
        logger_val.log_value('class acc', val_acc, epoch)               
        
        model_selected_loss = val_cls_loss + val_component_loss

        # Early stopping
        the_current_loss = val_loss
        if the_current_loss > the_last_loss:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!')
                with open(f"split_{opt.seed}_training_early_stop_epoch_{epoch}.txt", 'w') as f:
                    f.write(f'{epoch}')
                return
        else:
            trigger_times = 0
        the_last_loss = the_current_loss

        if best_loss > model_selected_loss:
            best_loss = model_selected_loss
            save_file = os.path.join(
                opt.save_folder, 'ckpt_best.pth'.format(epoch=epoch))
            save_model(model, optimizer_cls, optimizer_component, opt, epoch, save_file)
        

if __name__ == '__main__':
    main()
