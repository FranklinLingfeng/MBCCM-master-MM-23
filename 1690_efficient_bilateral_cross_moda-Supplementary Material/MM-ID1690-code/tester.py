import argparse
import easydict
import sys
import os
import os.path as osp
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.cluster import DBSCAN
import torch.distributed as dist


from tensorboardX import SummaryWriter
import random

import torch
from torch import nn
from torch.backends import cudnn
from utils.logging import Logger
from utils.serialization import copy_state_dict, load_checkpoint
from model.network import BaseResNet
from SYSU import SYSUMM01
from RegDB import RegDB


from dataset import TestData
from data_manager import *
from evaluator import test


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True

    main_worker(args)
    
    
    
def main_worker(args):
    
    cudnn.benchmark = True
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args)) 
      
    ## build dataset
    end = time.time()
    print("============load data==========")
    if args.dataset == 'sysu':
        data_dir = osp.join(args.data_dir, 'SYSU-MM01')
        dataset = SYSUMM01(args, args.data_dir)     
           
        query_img1, query_label1, query_cam1 = process_query_sysu(data_dir, mode='all')
        gall_img1, gall_label1, gall_cam1 = process_gallery_sysu(data_dir, mode='all', trial=0)
        
        gallset_all  = TestData(gall_img1, gall_label1, img_h=args.img_h, img_w=args.img_w)
        queryset_all = TestData(query_img1, query_label1, img_h=args.img_h, img_w=args.img_w)
        
        query_img2, query_label2, query_cam2 = process_query_sysu(data_dir, mode='indoor')
        gall_img2, gall_label2, gall_cam2 = process_gallery_sysu(data_dir, mode='indoor', trial=0)
        
        gallset_indoor  = TestData(gall_img2, gall_label2, img_h=args.img_h, img_w=args.img_w)
        queryset_indoor = TestData(query_img2, query_label2, img_h=args.img_h, img_w=args.img_w)
        
        gall_loader_all = data.DataLoader(gallset_all, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        query_loader_all = data.DataLoader(queryset_all, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        nquery_all = len(query_label1)
        ngall_all = len(gall_label1)
        
        gall_loader_indoor = data.DataLoader(gallset_indoor, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        query_loader_indoor = data.DataLoader(queryset_indoor, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        nquery_indoor = len(query_label2)
        ngall_indoor = len(gall_label2)
    
        print("  ----------------------------")
        print("  ALL SEARCH ")
        print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label1)), len(query_label1)))
        print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label1)), len(gall_label1)))
        print("  INDOOR SEARCH ")
        print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label2)), len(query_label2)))
        print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label2)), len(gall_label2)))
        print("  ----------------------------")
        
    elif args.dataset == 'regdb':
        
        ## parameters for regdb dataset
        args.eps = 0.3
        args.train_iter = 100
                
        data_dir = osp.join(args.data_dir, 'RegDB')
        dataset = RegDB(args, args.data_dir) 
        
        query_img1, query_label1 = process_test_regdb(data_dir, trial=1, modal='visible') ## query : visible 
        gall_img1, gall_label1 = process_test_regdb(data_dir, trial=1, modal='thermal') ## gallery : thermal
        
        query_img2, query_label2 = process_test_regdb(data_dir, trial=1, modal='thermal') ## query : visible 
        gall_img2, gall_label2 = process_test_regdb(data_dir, trial=1, modal='visible') ## gallery : thermal
        
        gallset_v2t  = TestData(gall_img1, gall_label1, img_h=args.img_h, img_w=args.img_w)
        queryset_v2t = TestData(query_img1, query_label1, img_h=args.img_h, img_w=args.img_w)
        
        gallset_t2v  = TestData(gall_img2, gall_label2, img_h=args.img_h, img_w=args.img_w)
        queryset_t2v = TestData(query_img2, query_label2, img_h=args.img_h, img_w=args.img_w)
        
        print("  ----------------------------")
        print("   visibletothermal   ")
        print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label1)), len(query_label1)))
        print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label1)), len(gall_label1)))
        print("   thermaltovisible   ")
        print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label2)), len(query_label2)))
        print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label2)), len(gall_label2)))
        print("  ----------------------------")
    
        # testing data loader
        gall_loader_v2t = data.DataLoader(gallset_v2t, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        query_loader_v2t = data.DataLoader(queryset_v2t, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        nquery_v2t = len(query_label1)
        ngall_v2t = len(gall_label1)
        
        gall_loader_t2v = data.DataLoader(gallset_t2v, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        query_loader_t2v = data.DataLoader(queryset_t2v, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        nquery_t2v = len(query_label2)
        ngall_t2v = len(gall_label2)
    
    ## model
    print('==> Building model..')
    main_net = BaseResNet(args, class_num=0, non_local='off', gm_pool='on', per_add_iters=args.per_add_iters)
    main_net.cuda()
    device_ids=[0, 1, 2]
    main_net = nn.DataParallel(main_net, device_ids=device_ids)
    ## load checkpoint
    if args.pretrained == True:
        if args.dataset == 'regdb':
            checkpoint = load_checkpoint(osp.join(args.regdb_model_dir, 'checkpoint.pth.tar'))
            copy_state_dict(checkpoint['state_dict'], main_net.module, strip='module.')
        elif args.dataset == 'sysu':
            checkpoint = load_checkpoint(osp.join(args.sysu_model_dir, 'checkpoint.pth.tar'))
            copy_state_dict(checkpoint['state_dict'], main_net.module, strip='module.')            

        ## evaluate

        if args.dataset == 'sysu':
            cmc_all, mAP_all, mINP_all = test(args, main_net,  
                                ngall_all, nquery_all, gall_loader_all, query_loader_all, 
                                query_label1, gall_label1, query_cam=query_cam1, gall_cam=gall_cam1)
            
            cmc_indoor, mAP_indoor, mINP_indoor = test(args, main_net,  
                                ngall_indoor, nquery_indoor, gall_loader_indoor, query_loader_indoor, 
                                query_label2, gall_label2, query_cam=query_cam2, gall_cam=gall_cam2)
            
            print('all:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_all[0], cmc_all[4], cmc_all[9], cmc_all[19], mAP_all, mINP_all))
            
            print('indoor:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_indoor[0], cmc_indoor[4], cmc_indoor[9], cmc_indoor[19], mAP_indoor, mINP_indoor))
                            
        elif args.dataset == 'regdb':
            cmc_v2t, mAP_v2t, mINP_v2t = test(args, main_net, 
                                    ngall_v2t, nquery_v2t,
                                    gall_loader_v2t, query_loader_v2t, 
                                    query_label1, gall_label1, test_mode=['IR', 'RGB'])
            
            cmc_t2v, mAP_t2v, mINP_t2v = test(args, main_net, 
                                    ngall_t2v, nquery_t2v,
                                    gall_loader_t2v, query_loader_t2v, 
                                    query_label2, gall_label2, test_mode=['RGB', 'IR'])
            
            print('VisibleToThermal:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_v2t[0], cmc_v2t[4], cmc_v2t[9], cmc_v2t[19], mAP_v2t, mINP_v2t))
            
            print('ThermalToVisible:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_t2v[0], cmc_t2v[4], cmc_t2v[9], cmc_t2v[19], mAP_t2v, mINP_t2v))

    
#### 
##############
##chennel argument / adapgray / randomchannel erasing

## 新的实验：
# 1、对于每个sample使用多个样本进行跨模态更新
# 2、使用标签修正


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="assignment main train")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    ## default
    parser.add_argument('--dataset', default='regdb', help='dataset name: regdb or sysu]') ### DATASET

    parser.add_argument('--epochs', default=80)
    parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
    parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
    parser.add_argument('--test-batch', default=256, type=int,
                    metavar='tb', help='testing batch size')
    parser.add_argument('--batch-size', default=12, type=int,
                    metavar='B', help='training batch size')
    parser.add_argument('--num_pos', default=8, type=int, 
                    help='num of pos per identity in each modality')
    parser.add_argument('--print-step', default=50, type=int)
    parser.add_argument('--eval-step', default=1, type=int)
    parser.add_argument('--start-epoch-two-modality', default=40, type=int)
    
    ## cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN") ## 0.6 for sysu and 0.3 for regdb
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    
    ## network  
    parser.add_argument('--arch', default='resnet50')
    parser.add_argument('--pool-dim', default=2048)
    parser.add_argument('--per-add-iters', default=1, help='param for GRL')
    parser.add_argument('--lr', default=0.00035, help='learning rate for main net')
    parser.add_argument('--optim', default='adam', help='optimizer')
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--train-iter', type=int, default=400) ## 200 for regdb and 400 for sysu
    parser.add_argument('--pretrained', type=bool, default=True)
    
    ## memory
    parser.add_argument('--momentum', type=float, default=0.1,
                        help="update momentum for the hybrid memory")
    parser.add_argument('--momentum-cross', type=float, default=0.1,
                        help="update momentum for the hybrid memory")
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    parser.add_argument('--use-hard', default=False)


    ## path
    
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/data/chengde/Lingfeng He/1_ReID_data')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs/test'))
    parser.add_argument('--sysu-model-dir', type=str, metavar='PATH',
                default=osp.join(working_dir, 'logs/model_sysu/best_model'))
    parser.add_argument('--regdb-model-dir', type=str, metavar='PATH',
                    default=osp.join(working_dir, 'logs/model_regdb/best_model'))
    main()