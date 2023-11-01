'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2023-01-16 14:03:09
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-02-06 00:00:50
FilePath: /Lingfeng He/xiongyali_new_idea/assignment/main_train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
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
from sklearn.cluster import DBSCAN
import torch.distributed as dist


import random
import collections
from trainer import mbccm_matching, get_match
from trainer import AssignTrainer
from sampler import IdentitySampler, IdentitySamplerAssign
from memory import ClusterMemory

import torch
from torch import nn
from torch.backends import cudnn
import torch.nn.functional as F
from utils.logging import Logger
from utils.serialization import save_checkpoint, copy_state_dict, load_checkpoint
from model.network import BaseResNet
from optimizer import select_optimizer
from utils.faiss_rerank import compute_jaccard_distance
from SYSU import SYSUMM01
from RegDB import RegDB


from dataset import pseudo_label_dataset, TestData, datasetr_for_feature_extractor
from data_manager import *
from evaluator import extract_features_for_cluster, test


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True

    main_worker(args)
    
    
def GenIdx( train_color_label, train_thermal_label):
    color_pos = collections.defaultdict(list)
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k,v in enumerate(train_color_label) if v==unique_label_color[i]]
        color_pos[unique_label_color[i]].extend(tmp_pos)
        
    thermal_pos = collections.defaultdict(list)
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k,v in enumerate(train_thermal_label) if v==unique_label_thermal[i]]
        thermal_pos[unique_label_thermal[i]].extend(tmp_pos)
    return color_pos, thermal_pos


## return dbscan centers
def dbscan_cluster(args, features1, features2, trainset, cluster):
    features1 = torch.cat([features1[f].unsqueeze(0) for f, _, _ in sorted(trainset)], 0)
    features2 = torch.cat([features2[f].unsqueeze(0) for f, _, _ in sorted(trainset)], 0)
    rerank_dist = compute_jaccard_distance(features1, k1=args.k1, k2=args.k2)
    print('----start DBScan clustering------')
    pseudo_labels = cluster.fit_predict(rerank_dist)
    num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
    
    centers1 = collections.defaultdict(list)
    centers2 = collections.defaultdict(list)
    num_ourliers = 0
    for i, label in enumerate(pseudo_labels):
        if label == -1:
            num_ourliers += 1
            continue
        centers1[pseudo_labels[i]].append(features1[i])
        centers2[pseudo_labels[i]].append(features2[i])
        
    print('----number of outliers: {:5d}'.format(num_ourliers))
    centers1 = [torch.stack(centers1[idx], dim=0).mean(0) for idx in sorted(centers1.keys())] 
    centers1 = torch.stack(centers1, dim=0)
    centers2 = [torch.stack(centers2[idx], dim=0).mean(0) for idx in sorted(centers2.keys())] 
    centers2 = torch.stack(centers2, dim=0)
    
    print('==> Statistics: {} clusters'.format(num_cluster))
    pseudo_dataset = []
    for i, ((fname, _, _), label) in enumerate(zip(sorted(trainset), pseudo_labels)): 
        if label != -1:
            pseudo_dataset.append((fname, label.item()))
    
    del rerank_dist
    
    return centers1, centers2, pseudo_labels, pseudo_dataset
    
    
    
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
    # device_ids=[0, 1, 2]
    # main_net = nn.DataParallel(main_net, device_ids=device_ids)
    main_net = nn.DataParallel(main_net)

    ## load checkpoint
    if args.pretrained == True:
        checkpoint = load_checkpoint(osp.join(args.logs_dir, 'baseline', 'checkpoint.pth.tar'))
        copy_state_dict(checkpoint['state_dict'], main_net.module, strip='module.')
        
    ## build optimizer and trainer
    optimizer = select_optimizer(args, main_net)

    trainer = AssignTrainer(args=args, encoder=main_net, batch_size=args.batch_size, num_pos=args.num_pos)
    trainer.temp = args.temp
    
    assign_RGB_set = datasetr_for_feature_extractor(dataset.train_rgb, img_h=args.img_h, img_w=args.img_w)
    assign_IR_set = datasetr_for_feature_extractor(dataset.train_ir, img_h=args.img_h, img_w=args.img_w)
    assign_RGB_loader = data.DataLoader(assign_RGB_set, batch_size=args.test_batch, num_workers=args.workers, drop_last=False)
    assign_IR_loader = data.DataLoader(assign_IR_set, batch_size=args.test_batch, num_workers=args.workers, drop_last=False)

    best_mAP = 0
    ## training
    
    if args.pretrained == True:
        num_epochs = args.epochs - args.start_epoch_two_modality
    else:
        num_epochs = args.epochs
    
    for epoch in range(num_epochs):
        
        if args.pretrained == True:
            epoch = epoch + args.start_epoch_two_modality
            if epoch == args.start_epoch_two_modality:
                eps = args.eps
                print('Clustering criterion: eps: {:.3f}'.format(eps))
                cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
            
        if epoch == 0:
            # DBSCAN cluster
            eps = args.eps
            print('Clustering criterion: eps: {:.3f}'.format(eps))
            cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

        ## extracting features and clustering
        print('==> start feature extracting...')
        features1_RGB, features2_RGB, _ = extract_features_for_cluster(main_net, assign_RGB_loader, mode='RGB')
        features1_IR, features2_IR, _ = extract_features_for_cluster(main_net, assign_IR_loader, mode='IR')  

        centers1_RGB, centers2_RGB, _, pseudo_dataset_RGB = dbscan_cluster(args, features1_RGB, features2_RGB, dataset.train_rgb, cluster)  
        centers1_IR, centers2_IR, _, pseudo_dataset_IR = dbscan_cluster(args, features1_IR, features2_IR, dataset.train_ir, cluster)  
        
        ## normolize
        centers1_RGB = F.normalize(centers1_RGB, dim=1)
        centers2_RGB = F.normalize(centers2_RGB, dim=1)
        
        centers1_IR = F.normalize(centers1_IR, dim=1)
        centers2_IR = F.normalize(centers2_IR, dim=1)
        
        ## assignment and build new dataset
        mask_matrix = mbccm_matching(centers1_RGB, centers1_IR, args, epoch)
        RGB_to_IR, IR_to_RGB, _, _ = get_match(mask_matrix)
        
        
        dataset_train = pseudo_label_dataset(args, pseudo_dataset_RGB, pseudo_dataset_IR, 
                                             img_h=args.img_h, img_w=args.img_w, epoch=epoch)   
        
        
        
        ##build memory
        memory_RGB = ClusterMemory(centers1_RGB.shape[1], centers1_RGB.shape[0], temp=args.temp,
                        momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_IR = ClusterMemory(centers1_IR.shape[1], centers1_IR.shape[0], temp=args.temp,
                        momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_IR_RGB = ClusterMemory(centers1_IR.shape[1], centers1_IR.shape[0], temp=args.temp,
                                    momentum=args.momentum_cross, use_hard=args.use_hard, cross_mode=False).cuda()
        memory_RGB_IR = ClusterMemory(centers1_RGB.shape[1], centers1_RGB.shape[0], temp=args.temp,
                                    momentum=args.momentum_cross, use_hard=args.use_hard, cross_mode=False).cuda()

        memory_RGB.features = centers1_RGB.cuda()
        memory_IR.features = centers1_IR.cuda()
        memory_RGB_IR.features = centers1_RGB.cuda()
        memory_IR_RGB.features = centers1_IR.cuda()
        trainer.memory_RGB = memory_RGB
        trainer.memory_IR = memory_IR
        trainer.memory_RGB_IR = memory_RGB_IR
        trainer.memory_IR_RGB = memory_IR_RGB
        
        ## build train loader
        color_pos, thermal_pos = GenIdx(dataset_train.label_RGB, dataset_train.label_IR)
        if epoch < args.start_epoch_two_modality:
            sampler = IdentitySampler(args, dataset_train.label_RGB, dataset_train.label_IR, 
                                    color_pos, thermal_pos, args.num_pos, args.batch_size, epoch)
        else:
            sampler = IdentitySamplerAssign(
                args, dataset_train.label_RGB, dataset_train.label_IR, 
                color_pos, thermal_pos, args.num_pos, args.batch_size,
                RGB_to_IR, IR_to_RGB, epoch
            )
            
        dataset_train.cIndex = sampler.index1  # color index
        dataset_train.tIndex = sampler.index2  # thermal index
        trainloader = data.DataLoader(dataset_train, batch_size=args.batch_size * args.num_pos, 
                                    sampler=sampler, num_workers=args.workers, drop_last=True)
        
        ## train
        print('==> start training...')
        trainer.train(args, epoch, trainloader, optimizer)

        ## evaluate
        if (epoch + 1) % args.eval_step == 0:
            print('Test Epoch: {}'.format(epoch))
            if args.dataset == 'sysu':
                cmc_all, mAP_all, mINP_all = test(args, main_net,  
                                    ngall_all, nquery_all, gall_loader_all, query_loader_all, 
                                    query_label1, gall_label1, query_cam=query_cam1, gall_cam=gall_cam1)
                
                cmc_indoor, mAP_indoor, mINP_indoor = test(args, main_net,  
                                    ngall_indoor, nquery_indoor, gall_loader_indoor, query_loader_indoor, 
                                    query_label2, gall_label2, query_cam=query_cam2, gall_cam=gall_cam2)
                
                cmc = cmc_all
                mAP = mAP_all
                mINP = mINP_all
                
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
                
                cmc = cmc_v2t
                mAP = mAP_v2t
                mINP = mINP_v2t
                
                print('VisibleToThermal:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc_v2t[0], cmc_v2t[4], cmc_v2t[9], cmc_v2t[19], mAP_v2t, mINP_v2t))
                
                print('ThermalToVisible:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc_t2v[0], cmc_t2v[4], cmc_t2v[9], cmc_t2v[19], mAP_t2v, mINP_t2v))

        # save model
            if mAP > best_mAP:  # not the real best for sysu-mm01
                best_mAP = mAP
                best_epoch = epoch
                state = {
                    'net': main_net.state_dict(),
                    'cmc': cmc,
                    'mAP': mAP,
                    'mINP': mINP,
                    'epoch': epoch,
                }
                torch.save(state, osp.join(args.logs_dir, (args.dataset+'_best.t')))

            print('Best Epoch [{}]'.format(best_epoch))    

            
        if epoch + 1 ==  args.start_epoch_two_modality:
            
            save_checkpoint({
                'state_dict': main_net.module.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best=True, fpath=osp.join(args.logs_dir, 'baseline', 'checkpoint.pth.tar'))
            
        if epoch + 1 ==  args.epochs:

            save_checkpoint({
                'state_dict': main_net.module.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best=True, fpath=osp.join(args.logs_dir, 
                                            'alpha' + str(args.alpha) + 'beta' + str(args.beta), 
                                            'checkpoint.pth.tar'))

    
#### 
##############


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="assignment main train")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    ## default
    parser.add_argument('--dataset', default='regdb', help='dataset name: regdb or sysu]') ### DATASET
    parser.add_argument('--mode', default='all', type=str, help='sysu:all/indoor regdb:visibletothermal')
    ### sysu: all indoor
    ### regdb: visibletothermal
    parser.add_argument('--epochs', default=80)
    parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
    parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
    parser.add_argument('--test-batch', default=256, type=int,
                    metavar='tb', help='testing batch size')
    parser.add_argument('--batch-size', default=12, type=int,
                    metavar='B', help='training batch size')
    parser.add_argument('--num_pos', default=12, type=int, 
                    help='num of pos per identity in each modality')
    parser.add_argument('--print-step', default=50, type=int)
    parser.add_argument('--eval-step', default=5, type=int)
    # parser.add_argument('--save-matrix-step', default=5, type=int)

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
    parser.add_argument('--train-iter', type=int, default=300) ## 100 for regdb and 300 for sysu
    parser.add_argument('--pretrained', type=bool, default=False)
    
    ## memory
    parser.add_argument('--momentum', type=float, default=0.1,
                        help="update momentum for the hybrid memory")
    parser.add_argument('--momentum-cross', type=float, default=0.1,
                        help="update momentum for the hybrid memory")
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    parser.add_argument('--use-hard', default=False)
    
    parser.add_argument('--alpha', default=0.9)
    parser.add_argument('--beta', default=0.5)



    ## path
    
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/data/chengde/Lingfeng He/1_ReID_data')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs/sysu'))
    main()

        
    
