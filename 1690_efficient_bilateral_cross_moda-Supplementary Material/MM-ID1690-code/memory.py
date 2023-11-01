'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2023-01-17 19:28:55
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2023-01-17 19:31:20
FilePath: /Lingfeng He/xiongyali_new_idea/assignment/memory.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd

class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum, cross_mode=False, threshold=0.5):
        
        ctx.cross_mode = cross_mode
        ctx.threshold = threshold
        
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        if ctx.cross_mode == False:
            for x, y in zip(inputs, targets):
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
                ctx.features[y] /= ctx.features[y].norm()
        else:
            for x, y in zip(inputs, targets):
                if torch.mm(ctx.features[y].unsqueeze(1).t(), x.unsqueeze(1)) > ctx.threshold:
                    ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
                    ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None, None, None


def cm(inputs, indexes, features, momentum=0.5, cross_mode=False, threshold=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device), cross_mode, threshold)


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False, cross_mode=False,
                 IR_to_RGB=None, RGB_to_IR=None):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.cross_mode = cross_mode
        self.threshold = 0.5

        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets):

        inputs = F.normalize(inputs, dim=1).cuda()
        
        outputs = cm(inputs, targets, self.features, self.momentum, cross_mode=self.cross_mode, threshold=self.threshold)
        outputs /= self.temp

        return outputs
    