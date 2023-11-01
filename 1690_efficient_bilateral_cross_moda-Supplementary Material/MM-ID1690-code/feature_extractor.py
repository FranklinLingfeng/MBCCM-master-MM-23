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
from evaluator import extract_features, extract_features_for_cluster, test
from dataset import datasetr_for_feature_extractor, TestData




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
        
    
    ## model
    print('==> Building model..')
    main_net = BaseResNet(args, class_num=0, non_local='off', gm_pool='on', per_add_iters=args.per_add_iters)
    main_net.cuda()
    device_ids=[0, 1, 2]
    main_net = nn.DataParallel(main_net, device_ids=device_ids)
    ## load checkpoint
    if args.pretrained == True:
        checkpoint = load_checkpoint(osp.join(args.sysu_model_dir, 'checkpoint.pth.tar'))
        copy_state_dict(checkpoint['state_dict'], main_net.module, strip='module.')   
        
    assign_RGB_set = datasetr_for_feature_extractor(dataset.train_rgb, img_h=args.img_h, img_w=args.img_w)
    assign_IR_set = datasetr_for_feature_extractor(dataset.train_ir, img_h=args.img_h, img_w=args.img_w)
    assign_RGB_loader = data.DataLoader(assign_RGB_set, batch_size=args.test_batch, num_workers=args.workers, drop_last=False)
    assign_IR_loader = data.DataLoader(assign_IR_set, batch_size=args.test_batch, num_workers=args.workers, drop_last=False)
        
    features1_RGB, features2_RGB, label_RGB = extract_features_for_cluster(main_net, assign_RGB_loader, mode='RGB')
    features1_IR, features2_IR, label_IR = extract_features_for_cluster(main_net, assign_IR_loader, mode='IR')  
             
    features1_RGB = torch.cat([features1_RGB[f].unsqueeze(0) for f, _, _, _ in sorted(assign_RGB_set)], 0)
    features1_IR = torch.cat([features1_IR[f].unsqueeze(0) for f, _, _, _ in sorted(assign_IR_set)], 0)      
    label_RGB = torch.cat([torch.tensor(label_RGB[f]).unsqueeze(0) for f, _, _, _ in sorted(assign_RGB_set)], 0)
    label_IR = torch.cat([torch.tensor(label_IR[f]).unsqueeze(0) for f, _, _, _ in sorted(assign_IR_set)], 0)  
    
    # np.save('/data/chengde/Lingfeng He/xiongyali_new_idea/assignment/feature_for_tsne/baseline/feature_RGB.npy', features1_RGB)
    # np.save('/data/chengde/Lingfeng He/xiongyali_new_idea/assignment/feature_for_tsne/baseline/feature_IR.npy', features1_IR)
    # np.save('/data/chengde/Lingfeng He/xiongyali_new_idea/assignment/feature_for_tsne/baseline/label_RGB.npy', label_RGB)
    # np.save('/data/chengde/Lingfeng He/xiongyali_new_idea/assignment/feature_for_tsne/baseline/label_IR.npy', label_IR)  
    
    np.save('/data/chengde/Lingfeng He/xiongyali_new_idea/assignment/feature_for_tsne/Hungarian/feature_RGB.npy', features1_RGB)
    np.save('/data/chengde/Lingfeng He/xiongyali_new_idea/assignment/feature_for_tsne/Hungarian/feature_IR.npy', features1_IR)
    np.save('/data/chengde/Lingfeng He/xiongyali_new_idea/assignment/feature_for_tsne/Hungarian/label_RGB.npy', label_RGB)
    np.save('/data/chengde/Lingfeng He/xiongyali_new_idea/assignment/feature_for_tsne/Hungarian/label_IR.npy', label_IR)
    
    # np.save('/data/chengde/Lingfeng He/xiongyali_new_idea/assignment/feature_for_tsne/improved_Hungarian/feature_RGB.npy', features1_RGB)
    # np.save('/data/chengde/Lingfeng He/xiongyali_new_idea/assignment/feature_for_tsne/improved_Hungarian/feature_IR.npy', features1_IR)
    # np.save('/data/chengde/Lingfeng He/xiongyali_new_idea/assignment/feature_for_tsne/improved_Hungarian/label_RGB.npy', label_RGB)
    # np.save('/data/chengde/Lingfeng He/xiongyali_new_idea/assignment/feature_for_tsne/improved_Hungarian/label_IR.npy', label_IR)
    
    # np.save('/data/chengde/Lingfeng He/xiongyali_new_idea/assignment/feature_for_tsne/improved_Hungarian_KL/feature_RGB.npy', features1_RGB)
    # np.save('/data/chengde/Lingfeng He/xiongyali_new_idea/assignment/feature_for_tsne/improved_Hungarian_KL/feature_IR.npy', features1_IR)
    # np.save('/data/chengde/Lingfeng He/xiongyali_new_idea/assignment/feature_for_tsne/improved_Hungarian_KL/label_RGB.npy', label_RGB)
    # np.save('/data/chengde/Lingfeng He/xiongyali_new_idea/assignment/feature_for_tsne/improved_Hungarian_KL/label_IR.npy', label_IR)

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
                            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="assignment main train")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    ## default
    parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]') ### DATASET

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
                        default=osp.join(working_dir, 'logs/main_train_regdb_100/test'))
    
    
    #SYSU
    parser.add_argument('--sysu-model-dir', type=str, metavar='PATH',
                default=osp.join(working_dir, 'logs/main_train/Hungarian_wo_improved'))
    
    ### REGDB
    parser.add_argument('--regdb-model-dir', type=str, metavar='PATH',
                    default=osp.join(working_dir, 'logs/main_train/Hungarian_wo_improved'))
    main()