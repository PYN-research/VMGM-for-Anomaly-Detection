from __future__ import absolute_import, print_function
import torch
from torch import nn
from torch.nn import functional as F
from memory_module import MemModule

class AutoEncoderFCMem(nn.Module):
    def __init__(self, in_col_dim, mem_dim, shrink_thres=0.0025):
        super(AutoEncoderFCMem, self).__init__()
        print('AutoEncoderFCMem')
        self.in_col_dim = in_col_dim
        self.mem_dim = mem_dim 
        feature_num = mem_dim//8  
        feature_num_2 = mem_dim//4  
        feature_num_x2 = mem_dim//2  
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.in_col_dim, out_features=feature_num, bias=True),
            nn.BatchNorm1d(num_features=feature_num),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(in_features=feature_num, out_features=feature_num_2, bias=True),
            nn.BatchNorm1d(num_features=feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(in_features=feature_num_2, out_features=feature_num_x2, bias=True),
            nn.BatchNorm1d(num_features=feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(in_features=feature_num_x2, out_features=feature_num_x2, bias=True),
            nn.BatchNorm1d(num_features=feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.mem_rep = MemModule(mem_dim=self.mem_dim, fea_dim=feature_num_x2, shrink_thres=shrink_thres)
        self.decoder = nn.Sequential(
            nn.Linear(in_features=feature_num_x2, out_features=feature_num_x2, bias=True),
            nn.BatchNorm1d(num_features=feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(in_features=feature_num_x2, out_features=feature_num_2, bias=True),
            nn.BatchNorm1d(num_features=feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(in_features=feature_num_2, out_features=feature_num, bias=True),
            nn.BatchNorm1d(num_features=feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(in_features=feature_num, out_features=self.in_col_dim, bias=True)
        )

    def forward(self, x):
        f = self.encoder(x)
        res_mem = self.mem_rep(f)
        f = res_mem['output']
        att = res_mem['att']
        output = self.decoder(f)
        return {'output': output, 'att': att, 'feat': f}

class MemAELoss(nn.Module):
    def __init__(self, regularization_parameter=0.0002):
        super(MemAELoss, self).__init__()
        self.reg_param = regularization_parameter

    def forward(self, prediction, ground_truth, observed_mask, training=False, testing=False, validating=False):
        attention_weights = prediction['att']
        observed_mask1 = observed_mask.type(torch.bool)
        masked_pred = torch.masked_select(prediction['output'],observed_mask1) #target_mask
        masked_true = torch.masked_select(ground_truth,observed_mask1)
        loss = None
        if training:
            loss = F.mse_loss(input=masked_true, target=masked_pred)
            regularizer =  F.softmax(attention_weights, dim=1) * F.log_softmax(attention_weights, dim=1)
            loss += (-1.0 * self.reg_param * regularizer.sum())
        if validating:
            loss = F.mse_loss(input=ground_truth, target=prediction['output'])
        if testing:
            loss = F.mse_loss(input=ground_truth, target=prediction['output'], reduction='none')
        return loss