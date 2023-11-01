'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2023-01-17 10:31:19
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-02-06 00:04:56
FilePath: /Lingfeng He/xiongyali_new_idea/assignment/sampler.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import random
import numpy as np
from torch.utils.data.sampler import Sampler
import sys
import os.path as osp
import torch
import copy



class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, args, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize, epoch):     

        uni_label_rgb = list(np.unique(train_color_label))
        uni_label_ir = list(np.unique(train_thermal_label))

            
        self.n_classes = len(uni_label_ir)
        self.iters = args.train_iter
        self.batchSize = batchSize
        self.num_pos = num_pos
        
        # N = np.maximum(len(train_color_label), len(train_thermal_label)) * self.iters_all
        N = self.batchSize * self.num_pos * self.iters
        self.N = N
        # uni_label_rgb_temp = copy.deepcopy(uni_label_rgb)
        # uni_label_ir_temp = copy.deepcopy(uni_label_ir)
        # index1 = list()
        # index2 = list()

        for j in range(self.iters):
        # for j in range(N // (self.num_pos * self.batchSize) + 1):
            batch_idx_rgb = np.random.choice(uni_label_rgb, batchSize, replace = False)
            batch_idx_ir = np.random.choice(uni_label_ir, batchSize, replace = False)
                
            for i in range(batchSize):
                if len(color_pos[batch_idx_rgb[i]]) > num_pos:
                    sample_color  = np.random.choice(color_pos[batch_idx_rgb[i]], num_pos, replace=False)
                else:
                    sample_color  = np.random.choice(color_pos[batch_idx_rgb[i]], num_pos, replace=True)

                if len(thermal_pos[batch_idx_ir[i]]) > num_pos:
                    sample_thermal = np.random.choice(thermal_pos[batch_idx_ir[i]], num_pos, replace=False)
                else:
                    sample_thermal = np.random.choice(thermal_pos[batch_idx_ir[i]], num_pos, replace=True)
                
                if j ==0 and i==0:
                    index1= sample_color
                    index2= sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))
        
        self.index1 = index1
        self.index2 = index2
        
    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N
    
    

class IdentitySamplerAssign(Sampler):
    def __init__(self, args, train_color_label, train_thermal_label,
                color_pos, thermal_pos, num_pos, batchSize, 
                RGB_to_IR, IR_to_RGB, epoch):
        # RGB_to_IR, IR_to_RGB
        
        self.iters = args.train_iter
        self.batchSize = batchSize
        self.num_pos = num_pos
        # uni_match_list = np.arange(len(match_list))
        
        uni_label_rgb = list(np.unique(train_color_label))
        uni_label_ir = list(np.unique(train_thermal_label))
        
        N = self.batchSize * self.num_pos * self.iters
        self.N = N
                
        for j in range(self.iters):
            
            # batch_idx_rgb = []
            # batch_idx_ir = []
            # match_idx = np.random.choice(uni_match_list, batchSize, replace=False)
            
            # for idx in match_idx:
            #     batch_idx_rgb.append(match_list[idx][1])
            #     batch_idx_ir.append(match_list[idx][0])
            
            # if j // 2 == 0:
            batch_idx_ir = np.random.choice(uni_label_ir, batchSize // 2, replace = False)
            batch_idx_rgb = []
                
            for idx in batch_idx_ir:
                batch_idx_rgb.append(int(np.random.choice(IR_to_RGB[idx], 1)))
                
            batch_idx_ir = np.array(batch_idx_ir)
            batch_idx_rgb = np.array(batch_idx_rgb)
            
            # else:
            batch_idx_rgb_add = np.random.choice(uni_label_rgb, batchSize // 2, replace = False)
            batch_idx_ir_add = []
                    
            for idx in batch_idx_rgb_add:
                batch_idx_ir_add.append(int(np.random.choice(RGB_to_IR[idx], 1)))
                
            batch_idx_rgb_add = np.array(batch_idx_rgb_add)
            batch_idx_ir_add = np.array(batch_idx_ir_add)
                
            batch_idx_ir = np.hstack((batch_idx_ir, batch_idx_ir_add))
            batch_idx_rgb = np.hstack((batch_idx_rgb, batch_idx_rgb_add))
            
            # batch_idx_rgb = np.random.choice(uni_label_rgb, batchSize, replace = False)
            # batch_idx_ir = []
                
            # for idx in batch_idx_rgb:
            #     batch_idx_ir.append(int(np.random.choice(RGB_to_IR[idx], 1)))
                
            # batch_idx_ir = np.array(batch_idx_ir)
            # batch_idx_rgb = np.array(batch_idx_rgb)
                    
            for i in range(self.batchSize):
                
                if len(color_pos[batch_idx_rgb[i]]) > num_pos:
                    sample_color  = np.random.choice(color_pos[batch_idx_rgb[i]], num_pos, replace=False)
                else:
                    sample_color  = np.random.choice(color_pos[batch_idx_rgb[i]], num_pos, replace=True)
                               
                if len(thermal_pos[batch_idx_ir[i]]) > num_pos:
                    sample_thermal = np.random.choice(thermal_pos[batch_idx_ir[i]], num_pos, replace=False)
                else:
                    sample_thermal = np.random.choice(thermal_pos[batch_idx_ir[i]], num_pos, replace=True)
                
                # sample_color = np.random.choice(color_pos[batch_idx_rgb[i]], 1)
                # sample_thermal = np.random.choice(thermal_pos[batch_idx_ir[i]], 1)
                    
                if j ==0 and i==0:
                    index1= sample_color
                    index2= sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))
        
        self.index1 = index1
        self.index2 = index2
        
    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N