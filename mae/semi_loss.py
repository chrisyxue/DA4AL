import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F


def EntropyLoss(predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-20):
    N, C = predict_prob.size()

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'
    
    predict_prob = F.softmax(predict_prob,dim=-1)
    entropy = -predict_prob * torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * entropy * class_level_weight) / float(N)


def L2RelationLoss(e1,feats,e1_unl,feats_unl):
    lab_num = e1.shape[0]
    unl_num = e1_unl.shape[0]
    e1_dis_m = torch.cdist(e1.unsqueeze(0),e1_unl.unsqueeze(0)).squeeze(0)
    feats_dis_m = torch.cdist(feats.unsqueeze(0),feats_unl.unsqueeze(0)).squeeze(0)
    loss = torch.norm(feats_dis_m - e1_dis_m,dim=-1)/unl_num 
    loss = loss.mean()
    return loss


def L2RelationNormLoss(e1,feats,e1_unl,feats_unl):
    lab_num = e1.shape[0]
    unl_num = e1_unl.shape[0]
    e1_dis_m = torch.cdist(e1.unsqueeze(0),e1_unl.unsqueeze(0)).squeeze(0)
    feats_dis_m = torch.cdist(feats.unsqueeze(0),feats_unl.unsqueeze(0)).squeeze(0)
    e1_dis_m = F.softmax(e1_dis_m,dim=-1)
    feats_dis_m = F.softmax(feats_dis_m,dim=-1)
    loss = torch.norm(feats_dis_m - e1_dis_m,dim=-1)
    loss = loss.mean()
    return loss