'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2023-01-17 09:53:38
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-02-06 00:26:03
FilePath: /Lingfeng He/xiongyali_new_idea/assignment/assign.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from scipy.optimize import linear_sum_assignment
from utils.meters import AverageMeter
from optimizer import adjust_learning_rate
import time
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import collections
import torchvision.transforms as transforms
from PIL import Image
import copy
import random

import torch.nn.functional as F

from utils.faiss_rerank import rerank_jaccard


# def direct_match(centers_RGB, centers_IR):
    
#     num_center_RGB = centers_RGB.shape[0]
#     num_center_IR = centers_IR.shape[0]
#     dist_mat = pairwise_distance(centers_IR, centers_RGB)
#     mask_matrix_RGB = torch.zeros_like(dist_mat)
#     mask_matrix_IR = torch.zeros_like(dist_mat)

#     # for i in range(num_center_IR):
#     #     index_max = torch.argmin(dist_mat[i])
#     #     mask_matrix[i, index_max] = 1
#     for i in range(num_center_RGB): ## torch.sum dim=1
#         index_max = torch.argmax(dist_mat[:, i])
#         mask_matrix_RGB[index_max, i] = 1
    
#     for i in range(num_center_IR): ## dim=0
#         index_max = torch.argmax(dist_mat[i])
#         mask_matrix_IR[i, index_max] = 1
#     return mask_matrix_RGB, mask_matrix_IR


def mbccm_matching(centers_RGB, centers_IR, args, epoch):
    
    num_center_RGB = centers_RGB.shape[0]
    num_center_IR = centers_IR.shape[0]
    
    if num_center_RGB >= num_center_IR:
        
        k = num_center_RGB // num_center_IR + 1
        
        dist_mat = pairwise_distance(centers_IR, centers_RGB)
                
        cost_mat = dist_mat.repeat(k, 1)

        IR_idx1, RGB_idx1 = linear_sum_assignment(dist_mat) ## ir -> rgb
        IR_idx2, RGB_idx2 = linear_sum_assignment(cost_mat) ## repeat rgb - to - ir
            
    elif num_center_RGB < num_center_IR:
        
        k = num_center_IR // num_center_RGB + 1    
        
        dist_mat = pairwise_distance(centers_RGB, centers_IR)
                
        cost_mat = dist_mat.repeat(k, 1)

        RGB_idx1, IR_idx1 = linear_sum_assignment(dist_mat) ## ir -> rgb
        RGB_idx2, IR_idx2 = linear_sum_assignment(cost_mat) ## repeat rgb - to - ir
        dist_mat = dist_mat.T
        
    
    RGB_to_IR = collections.defaultdict(list)
    IR_to_RGB = collections.defaultdict(list)
    mask_matrix1 = np.array(torch.zeros_like(dist_mat))
    mask_matrix2 = np.array(torch.zeros_like(dist_mat)) 

    for j in range(len(IR_idx1)):
        idx = IR_idx1[j] % num_center_IR
        index_small = list(np.where(dist_mat[idx, :] <= dist_mat[idx, RGB_idx1[j] % num_center_RGB])[0])
        mask_matrix1[idx, index_small] = 1
        
    for j in range(len(RGB_idx2)):
        idx = RGB_idx2[j] % num_center_RGB
        index_small = list(np.where(dist_mat[:, idx] <= dist_mat[IR_idx2[j] % num_center_IR, idx]))
        mask_matrix2[index_small, idx] = 1
        
    mask_matrix = 1 - (1 - mask_matrix1) * (1 - mask_matrix2)
    
    return mask_matrix



def get_match(mask_matrix):
    
    num_IR_mean = np.mean(np.sum(mask_matrix, axis=1))
    num_RGB_mean = np.mean(np.sum(mask_matrix, axis=0))
    print('mean number of neighbors for IR and RGB :{:.5f} /// {:.5f}'.format(num_IR_mean, num_RGB_mean))

    RGB_to_IR = collections.defaultdict(list)
    IR_to_RGB = collections.defaultdict(list)
    
    for i in range(mask_matrix.shape[0]):
        IR_to_RGB[i].extend(list(np.where(mask_matrix[i, :] == 1)[0]))
        
    for j in range(mask_matrix.shape[1]):
        RGB_to_IR[j].extend(list(np.where(mask_matrix[:, j] == 1)[0]))

    return RGB_to_IR, IR_to_RGB, num_IR_mean, num_RGB_mean


        

def pairwise_distance(x, y):
   
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_mat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(x, y.t(), beta=1, alpha=-2)
        
    return dist_mat


class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()

    def forward(self, pred, label):

        predict = F.softmax(pred, dim=1)
        target = F.softmax(label, dim=1)
        
        predict = predict.clamp(min=1e-12)
        target = target.clamp(min=1e-12)
        
        loss1 = ((target * (target.log() - predict.log())).sum(1).sum() / target.size()[0])
        loss2 = ((predict * (predict.log() - target.log())).sum(1).sum() / target.size()[0])

        loss = (loss1 + loss2) / 2
        
        return loss


class AssignTrainer(object):
    def __init__(self, args, encoder, batch_size, num_pos, memory_RGB=None, memory_IR=None, memory_RGB_IR=None, memory_IR_RGB=None,
                 RGB_to_IR=None, IR_to_RGB=None):
        super(AssignTrainer, self).__init__()
        self.encoder = encoder
        
        self.memory_RGB = memory_RGB
        self.memory_IR = memory_IR
        self.memory_RGB_IR = memory_RGB_IR
        self.memory_IR_RGB = memory_IR_RGB
        
        self.temp = 0.05
        
        self.mask_matrix = None
        
        self.criterion_dis = nn.BCELoss()
        
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        self.transform_to_image = transforms.Compose([
             transforms.ToPILImage()
        ])
        
        self.RGB_to_IR = RGB_to_IR
        self.IR_to_RGB = IR_to_RGB
        
        self.alpha = args.alpha
        self.beta = args.beta
        
        self.match_acc = []
        
        print('alpha {:.3f} // beta:{:.3f}'.format(self.alpha, self.beta))
            
        self.kl_div_loss = KLDivLoss()
                
    def train(self, args, epoch, trainloader, optimizer):
        current_lr = adjust_learning_rate(args, optimizer, epoch)
        contrast_loss = AverageMeter()

        batch_time = AverageMeter()
        loss1_loss = AverageMeter()
        loss2_loss = AverageMeter()
        loss3_loss = AverageMeter()
        loss4_loss = AverageMeter()
        loss5_loss = AverageMeter()
        loss6_loss = AverageMeter()
        match_accuracy = AverageMeter()

        
        # switch to train mode
        self.encoder.train()
        end = time.time()
                        
        print('epoch:{:5d}'.format(epoch))

        for batch_idx, (img1_0, img1_1, img2, label1, label2) in enumerate(trainloader):
        
            input10 = Variable(img1_0.cuda())
            input11 = Variable(img1_1.cuda())
            input2 = Variable(img2.cuda())    

            label1 = Variable(label1.cuda())
            label2 = Variable(label2.cuda())
                        
            input1 = torch.cat((input10, input11), dim=0)
            feat_rgb, feat_ir, x_pool_rgb, x_pool_ir = self.encoder(input1, input2)
            
            feat_rgb_1 = feat_rgb[:feat_rgb.shape[0] // 2]
            feat_rgb_2 = feat_rgb[feat_rgb.shape[0] // 2:]
            
            
            output1 = self.memory_RGB(feat_rgb_1, label1)
            output2 = self.memory_RGB(feat_rgb_2, label1)
            output3 = self.memory_IR(feat_ir, label2)
            loss1 = F.cross_entropy(output1, label1)
            loss2 = F.cross_entropy(output2, label1)
            # loss12 = F.cross_entropy(torch.cat((output1, output2), dim=0), torch.cat((label1, label1), dim=0))
            loss3 = F.cross_entropy(output3, label2)
            
            output_rgb_cross1 = torch.mm(feat_rgb_1, self.memory_IR.features.t()) / self.temp
            output_rgb_cross2 = torch.mm(feat_rgb_2, self.memory_IR.features.t()) / self.temp
            output_ir_cross = torch.mm(feat_ir, self.memory_RGB.features.t()) / self.temp
            
            if epoch < args.start_epoch_two_modality:       
                loss_ms = loss2 + loss1 + loss3
                # loss_contrast = loss12 + loss3
                loss_all = loss_ms
                
            else:     
                output_rgb_rgb_1 = self.memory_RGB_IR(feat_rgb_1, label1) 
                output_rgb_rgb_2 = self.memory_RGB_IR(feat_rgb_2, label1)
                output_ir_rgb = self.memory_RGB_IR(feat_ir, label1)
                
                output_rgb_ir_1 = self.memory_IR_RGB(feat_rgb_1, label2) 
                output_rgb_ir_2 = self.memory_IR_RGB(feat_rgb_2, label2) 
                output_ir_ir = self.memory_IR_RGB(feat_ir, label2)

                loss_ma_rgb = F.cross_entropy(output_rgb_rgb_1, label1) + F.cross_entropy(output_rgb_rgb_2, label1) \
                    + F.cross_entropy(output_ir_rgb, label1)
                    
                loss_ma_ir = F.cross_entropy(output_rgb_ir_1, label2) + F.cross_entropy(output_rgb_ir_2, label2) \
                    + F.cross_entropy(output_ir_ir, label2)                        

                loss_cc = self.kl_div_loss(output1, output_rgb_rgb_1) \
                    + self.kl_div_loss(output2, output_rgb_rgb_2) \
                    + self.kl_div_loss(output_rgb_ir_1, output_rgb_cross1) \
                    + self.kl_div_loss(output_rgb_ir_2, output_rgb_cross2) \
                    + self.kl_div_loss(output_ir_rgb, output_ir_cross) \
                    + self.kl_div_loss(output3, output_ir_ir)
                                          
                loss4_loss.update(loss_ma_rgb.item())
                loss5_loss.update(loss_ma_ir.item())
                
                loss_ms = loss1 + loss2 + loss3
                loss_all = loss_ms + self.alpha * (loss_ma_rgb + loss_ma_ir) + self.beta * loss_cc
                                
                loss6_loss.update(loss_cc.item()) 
                
                
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            contrast_loss.update(loss_ms.item())
            loss1_loss.update(loss1.item())
            loss2_loss.update(loss2.item())
            loss3_loss.update(loss3.item())
            

            # print log
            batch_time.update(time.time() - end)
            end = time.time()
            
            if (batch_idx + 1) % args.print_step == 0:
                if epoch < args.start_epoch_two_modality: 
                    print('Epoch: [{}][{}/{}] '
                        'lr:{:.8f} '
                        'MSLoss: {contrast_loss.val:.4f}({contrast_loss.avg:.3f}) '
                        'RGBLoss: {loss1_loss.val:.4f}({loss1_loss.avg:.3f}) '
                        'RGBCALoss: {loss2_loss.val:.4f}({loss2_loss.avg:.3f}) '
                        'IRLoss: {loss3_loss.val:.4f}({loss3_loss.avg:.3f}) '.format(
                                                        epoch, batch_idx, len(trainloader), current_lr,
                                                        contrast_loss = contrast_loss,
                                                        loss1_loss = loss1_loss,
                                                        loss2_loss = loss2_loss,
                                                        loss3_loss = loss3_loss))
                else:
                    print('Epoch: [{}][{}/{}] '
                        'lr:{:.8f} '
                        'MSloss: {contrast_loss.val:.4f}({contrast_loss.avg:.3f}) '
                        'RGB-MSLoss: {loss1_loss.val:.4f}({loss1_loss.avg:.3f}) '
                        'RGB-CA-MSLoss: {loss2_loss.val:.4f}({loss2_loss.avg:.3f}) '
                        'IR-MSLoss: {loss3_loss.val:.4f}({loss3_loss.avg:.3f}) '
                        '\nRGB-MAloss: {loss4_loss.val:.4f}({loss4_loss.avg:.3f})'
                        'IR-MAloss: {loss5_loss.val:.4f}({loss5_loss.avg:.3f})'
                        'CC_loss: {loss6_loss.val:.4f}({loss6_loss.avg:.3f})'.format(
                                                        epoch, batch_idx, len(trainloader), current_lr,
                                                        contrast_loss = contrast_loss,
                                                        loss1_loss = loss1_loss,
                                                        loss2_loss = loss2_loss,
                                                        loss3_loss = loss3_loss,
                                                        loss4_loss = loss4_loss,
                                                        loss5_loss = loss5_loss,
                                                        loss6_loss = loss6_loss))               
            
            
            