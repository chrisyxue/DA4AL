from functools import partial

import torch
import torch.nn as nn
import pdb
from timm.models.vision_transformer import Block
import torch.nn.functional as F
import os
import torchshow as ts
from captum.attr import (
    GuidedGradCam,
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
    Saliency,
    Occlusion,
    LRP
)


class ResMLP(nn.Module):
    def __init__(self, embed_dim, reduction):
        super(ResMLP, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim // reduction)
        self.bn1 = nn.BatchNorm1d(embed_dim // reduction)
        self.fc2 = nn.Linear(embed_dim // reduction, embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)
        
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
    
    def forward(self,x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = x + out
        return out
        

class Adapter(nn.Module):
    """
    Adapter for MAE For AL
    """
    def __init__(self, embed_dim=1024,num_classes=10,reduction=2,adapter_type='MLP'):
        super(Adapter,self).__init__()
        self.embed_dim = embed_dim

        if adapter_type == 'MLP':
            self.adapter = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // reduction),
                nn.BatchNorm1d(embed_dim // reduction),
                nn.ReLU(inplace=False),
                nn.Linear(embed_dim // reduction, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(inplace=False)
            )
        elif adapter_type == 'ResMLP':
            self.adapter = ResMLP(embed_dim, reduction)
        else:
            raise ValueError('No Adapter like:'+str(adapter_type))
        
        self.cls = nn.Linear(embed_dim,num_classes,bias=False)
    
    def forward_emb(self,x):
        """
         x -> [batch_size, dim]
        """
        out = self.adapter(x)
        return out
    
    def forward(self,x):
        """
         x -> [batch_size, dim]
        """
        emb = self.forward_emb(x)
        out = self.cls(emb)
        return out,emb
    
    def forward_out(self,x):
        """
         x -> [batch_size, dim]
        """
        emb = self.forward_emb(x)
        out = self.cls(emb)
        return out
    
    def forward_all(self,x):
        """
         x-> [batch_size, patch_num, dim]
        """
        batch_size = x.shape[0]
        patch_num = x.shape[1]
        x = x.reshape([batch_size*patch_num,-1])
        x = self.adapter(x)
        out = x.reshape([batch_size,patch_num,-1])
        return out
    
    def forward_att(self,x,top_ratio=0.5,use_global_avg=False,metric='cosine'):
        """
        x -> [batch_size, patch_num, dim]

        return: the top_k for label-irrelevant patches
        """
        x = self.forward_all(x)

        # get attention map
        patches = x[:, 1:, :]
        if use_global_avg == False:
            cls = x[:, 0].unsqueeze(1)
        else:
            cls = x[:, 1:, :].mean(dim=1).unsqueeze(1)
        
        cls_ext = cls.repeat([1,patches.shape[1],1])
        att = 1 - torch.cosine_similarity(patches,cls_ext,dim=-1) # [batch_size, patch_num]

        # get top_k
        top_k = int(att.shape[-1]*top_ratio)  
        # select
        _, topk_idx = torch.topk(att,top_k,dim=-1)
        topk_idx = topk_idx.cpu().numpy().tolist()
        return att,topk_idx

    def get_embedding_dim(self):
        return self.embed_dim