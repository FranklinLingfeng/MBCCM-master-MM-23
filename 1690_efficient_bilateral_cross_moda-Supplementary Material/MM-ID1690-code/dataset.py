'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2023-01-16 14:38:24
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-02-04 15:11:08
FilePath: /Lingfeng He/xiongyali_new_idea/assignment/dataset.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import os.path as osp
import pickle
import torchvision.transforms as transforms
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing, LinearTransform
import random
import torch.utils.data as data
from PIL import Image
import math
import torch

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class ChannelExchange(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """

    def __init__(self, gray=3, probability=1):
        self.gray = gray
        self.probability = probability

    def __call__(self, img):

        idx = random.randint(0, self.gray)

        if idx == 0:
            # random select R Channel
            img[1, :, :] = img[0, :, :]
            img[2, :, :] = img[0, :, :]
        elif idx == 1:
            # random select B Channel
            img[0, :, :] = img[1, :, :]
            img[2, :, :] = img[1, :, :]
        elif idx == 2:
            # random select G Channel
            img[0, :, :] = img[2, :, :]
            img[1, :, :] = img[2, :, :]
            
        else:
            if random.uniform(0, 1) > self.probability:
                # return img
                img = img
            else:
                tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
                img[0, :, :] = tmp_img
                img[1, :, :] = tmp_img
                img[2, :, :] = tmp_img
        return img
    
    
class togrey(object):
    
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):     
        
        tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
        img[0, :, :] = tmp_img
        img[1, :, :] = tmp_img
        img[2, :, :] = tmp_img   
        
        return img
    


## train set
class pseudo_label_dataset(data.Dataset):
    
    def __init__(self, args, RGB_set=None, IR_set=None, IR_to_RGB=None, RGB_to_IR=None, 
                 data_path_RGB=None, data_path_IR=None, colorIndex = None, thermalIndex = None, 
                 img_h=288, img_w=144, have_pseudo_labels=False, epoch=0):
        
        self.epoch=epoch
        self.start_two_modality = args.start_epoch_two_modality
        
        self.have_pseudo_labels = have_pseudo_labels
        self.file_IR = []
        self.label_IR = []
        self.file_RGB = []
        self.label_RGB = []          
        print('load data---')
    
        for fname, label in RGB_set:
            self.file_RGB.append(fname)
            self.label_RGB.append(label)
        for fname, label in IR_set:
            self.file_IR.append(fname)
            self.label_IR.append(label)
        train_image_rgb = []
        
        for i in range(len(self.file_RGB)):
            img = Image.open(self.file_RGB[i])
            img = img.resize((img_w, img_h), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_image_rgb.append(pix_array)
        train_image_rgb = np.array(train_image_rgb)
        
        train_image_ir = []
        for i in range(len(self.file_IR)):
            img = Image.open(self.file_IR[i])
            img = img.resize((img_w, img_h), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_image_ir.append(pix_array)
        train_image_ir = np.array(train_image_ir)
        
        self.train_image_rgb = train_image_rgb
        self.train_image_ir = train_image_ir
        
        print('Finish loading data---')
        
        ## image transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.transform_thermal = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Resize((img_h, img_w)),
            transforms.RandomHorizontalFlip(),
            transforms.Pad(10),
            transforms.RandomCrop((img_h, img_w)),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ChannelAdapGray(probability =0.5)])
            
        self.transform_color = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Resize((img_h, img_w)),
            transforms.RandomHorizontalFlip(),
            transforms.Pad(10),
            transforms.RandomCrop((img_h, img_w)),
            transforms.ToTensor(),
            LinearTransform(probability = 0.8),
            normalize,
            ChannelRandomErasing(probability = 0.5)])
            
        self.transform_color1 = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Resize((img_h, img_w)),
            transforms.RandomHorizontalFlip(),
            transforms.Pad(10),
            transforms.RandomCrop((img_h, img_w)),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ChannelExchange(gray=2)])
        
        
        self.cIndex = colorIndex
        self.tIndex = thermalIndex    
            
        
    def __getitem__(self, index):
        
        img1 = self.train_image_rgb[self.cIndex[index]]   
        img1_label = self.label_RGB[self.cIndex[index]]
        img2 = self.train_image_ir[self.tIndex[index]]
        img2_label = self.label_IR[self.tIndex[index]]

        # if self.epoch < self.start_two_modality:
        img1_0 = self.transform_color(img1)
        img1_1 = self.transform_color1(img1)
        img2 = self.transform_thermal(img2) 
        
        return img1_0, img1_1, img2, img1_label, img2_label
        
        
    def _build_dataset(self, data_path):
        files = []
        labels = []
        num_outliers = 0
        if not osp.exists(data_path):
            raise RuntimeError("'{}' is not available".format(data_path))  
        with open(data_path, 'rb') as f:
            pseudo_label = pickle.load(f)
            
        for file in pseudo_label.keys():
            label = int(pseudo_label[file])
            if label != -1:
                files.append(file)
                labels.append(label)
            else:
                num_outliers += 1
        
        num_clusters = len(np.unique(labels))
        print('length of dataset:{:5d}\nnumber of clusters:{:5d}\nnumber of outliers:{:5d}'
              .format(int(len(files)), int(num_clusters), int(num_outliers)))
        
        return files, labels
    
    
class datasetr_for_feature_extractor(data.Dataset):
    
    def __init__(self, trainset, img_h=288, img_w=144):   
        
        
        cluster_image = []
        label = []
        
        self.trainset = trainset
        
        for i in range(len(trainset)):
            img = Image.open(trainset[i][0])
            img = img.resize((img_w, img_h), Image.ANTIALIAS)
            pix_array = np.array(img)
            cluster_image.append(pix_array)
            label.append(int(trainset[i][1]))
        cluster_image = np.array(cluster_image)
        self.cluster_image = cluster_image
        self.label = label
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_test = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
            normalize])
        
        self.transform_intermediate = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
            normalize,
            togrey(probability=1)])
        
    def __getitem__(self, index):
        img = self.cluster_image[index]
        img_original = self.transform_test(img)
        img_intermeidate = self.transform_intermediate(img)
        label = self.label[index]
        img_path = self.trainset[index][0]

        return img_path, img_original, img_intermeidate, label

    def __len__(self):
        return len(self.trainset)
    

        
        
## test set
class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, img_h=288, img_w=144):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_w, img_h), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
            normalize])

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)