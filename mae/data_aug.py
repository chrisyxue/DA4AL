from functools import partial

import torch
import torch.nn as nn
import pdb
from timm.models.vision_transformer import Block
import torch.nn.functional as F
import os
import torchshow as ts
from torchvision import transforms
from timm.data.mixup import Mixup
# from active_learning.Extremal_Perturbation import extremal_perturbation_batch,contrastive_reward
# from active_learning.Data_Augmentation import NormalAug, SelfAug, AutoAug, RandAug, MixUp, CutMix, CutMixUp, ManifoldMixUp
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

def get_tensor_augdata(args_input,args_task,images,labels,model_mae,adapter,sample_aug_data_batch):
    if args_input.aug_strategy == 'MixUp':
        imgs_aug,labels_aug = MixUp(args_task,images,labels)
    elif args_input.aug_strategy == 'CutMix':
        imgs_aug,labels_aug = CutMix(args_task,images,labels)
    elif args_input.aug_strategy == 'CutMixUp':
        imgs_aug,labels_aug = CutMixUp(args_task,images,labels)
    elif args_input.aug_strategy == 'Self' or 'Sample':
        imgs_aug,labels_aug = Local_Aug(args_input,args_task,images,labels,model_mae,adapter,sample_aug_data_batch)
    else:
        raise ValueError('No such aug strategy:'+str(args_input.aug_strategy))
    return imgs_aug,labels_aug


class CombineModel(nn.Module):
    """
    Combine Adapter and MAE
    """

    def __init__(self, model, adapter,args):
        super(CombineModel,self).__init__()
        self.model = model
        self.adapter = adapter
        self.use_global_avg = args.use_global_avg
    
    def forward(self,x):
        out,_,_ = self.model.forward_encoder(x,0)
        if self.use_global_avg:
            out = out[:, 1:, :].mean(dim=1)
        else:
            out = out[:, 0]
        out,rep = self.adapter(out)
        return out


"""
Augmentations in the transform function
"""

def get_augmentor(args_input,args_task):
    # print(args_task)
    if 'is_gray' in args_task.keys():
        is_gray = args_task['is_gray']
    else:
        is_gray = False
    print(is_gray)
    if args_input.aug_strategy == 'Normal':
        if is_gray:
            augmentor = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomHorizontalFlip(p=0.9),
                transforms.RandomVerticalFlip(p=0.9),
                transforms.Resize(args_task['img_size']),
                transforms.ToTensor(),
                transforms.Normalize(mean=args_task['mean'], std=args_task['std'])
            ])
        else:
            augmentor = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.9),
                transforms.RandomVerticalFlip(p=0.9),
                transforms.Resize(args_task['img_size']),
			    transforms.ToTensor(),
			    transforms.Normalize(mean=args_task['mean'], std=args_task['std'])
            ])

    elif args_input.aug_strategy == 'Auto':
        if is_gray:
            augmentor = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
                transforms.Resize(args_task['img_size']),
                transforms.ToTensor(),
                transforms.Normalize(mean=args_task['mean'], std=args_task['std'])
            ])
        else:
            augmentor = transforms.Compose([
                transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
                transforms.Resize(args_task['img_size']),
                transforms.ToTensor(),
                transforms.Normalize(mean=args_task['mean'], std=args_task['std'])
            ])
    
    elif args_input.aug_strategy == 'Rand':
        if is_gray:
            augmentor = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.RandAugment(num_ops=5),
                transforms.Resize(args_task['img_size']),
                transforms.ToTensor(),
                transforms.Normalize(mean=args_task['mean'], std=args_task['std'])
            ])
        else:
             augmentor = transforms.Compose([
                transforms.RandAugment(num_ops=5),
                transforms.Resize(args_task['img_size']),
                transforms.ToTensor(),
                transforms.Normalize(mean=args_task['mean'], std=args_task['std'])
            ])
    else:
        augmentor = None
    return augmentor


"""
 Augmentations applied to the tensor
"""

MIXUP_ARGS = {
    'mixup_alpha': 1.,
    'cutmix_alpha': 0.,
    'cutmix_minmax': None,
    'prob': 1.0,
    'switch_prob': 0.,
    'mode': 'elem',
    'label_smoothing': 0,
    'num_classes': 10}

CUTMIX_ARGS = {
    'mixup_alpha': 0.,
    'cutmix_alpha': 1.0,
    'cutmix_minmax': None,
    'prob': 1.0,
    'switch_prob': 0.,
    'mode': 'elem',
    'label_smoothing': 0,
    'num_classes': 10}

CUTMIXUP_ARGS = {
    'mixup_alpha': 0.3,
    'cutmix_alpha': 0.7,
    'cutmix_minmax': None,
    'prob': 1.0,
    'switch_prob': 0,
    'mode': 'elem',
    'label_smoothing': 0,
    'num_classes': 1000}

def ManifoldMixUp(args_task,images,labels,model):
    # MIXUP_ARGS['num_classes'] = args_task.num_class
    MIXUP_ARGS['num_classes'] = args_task['num_class']
    batch_size = images.shape[0]
    if batch_size % 2 != 0:
        images = images[:-1]
        labels = labels[:-1]
    feats,_,_ = model.forward_encoder(images,0)
    feats = model.unpatchify(feats[:, 1:, :])
    mixup_fn = Mixup(**MIXUP_ARGS)
    feats,labels = mixup_fn(feats,labels)
    feats = model.patchify(feats).mean(dim=1)
    return feats,labels

def MixUp(args_task,images,labels):
    # MIXUP_ARGS['num_classes'] = args_task.num_class
    MIXUP_ARGS['num_classes'] = args_task['num_class']
    # make the batch_size be even
    batch_size = images.shape[0]
    if batch_size % 2 != 0:
        images = images[:-1]
        labels = labels[:-1]
    mixup_fn = Mixup(**MIXUP_ARGS)
    imgs_aug,labels = mixup_fn(images,labels)
    return imgs_aug,labels

def CutMix(args_task,images,labels):
    # CUTMIX_ARGS['num_classes'] = args_task.num_class
    CUTMIX_ARGS['num_classes'] = args_task['num_class']
    # make the batch_size be even
    batch_size = images.shape[0]
    if batch_size % 2 != 0:
        images = images[:-1]
        labels = labels[:-1]
    mixup_fn = Mixup(**CUTMIX_ARGS)
    imgs_aug,labels = mixup_fn(images,labels)
    return imgs_aug,labels

def CutMixUp(args_task,images,labels):
    #  CUTMIXUP_ARGS['num_classes'] = args_task.num_class
    CUTMIXUP_ARGS['num_classes'] = args_task['num_class']
    # make the batch_size be even
    batch_size = images.shape[0]
    if batch_size % 2 != 0:
        images = images[:-1]
        labels = labels[:-1]
    mixup_fn = Mixup(**CUTMIXUP_ARGS)
    imgs_aug,labels = mixup_fn(images,labels)
    return imgs_aug,labels


def SelfAug(images,feats,model,ids_restore,mask):
    model.zero_grad()
    pred = model.forward_decoder(feats, ids_restore)                
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    # loss = self.forward_loss(imgs, pred, mask)
    mask = model.unpatchify(mask)
    pred = model.unpatchify(pred)
    imgs_aug = images * (1 - mask) + pred * mask
    return imgs_aug

def Label_Smooth(labels,epsilons):
    """
    labels -> [batch,class]
    epsilons -> [batch]
    """
    return labels * (1-epsilons.unsqueeze(1)) + epsilons.unsqueeze(1)/labels.shape[1]

def SamplePatchAug(model_mae,args_input,sample_aug_data_batch, mask, images):
    sample_aug_data_batch = torch.concat(sample_aug_data_batch)
    batch_size = sample_aug_data_batch.shape[0]
    aug_data_num = sample_aug_data_batch.shape[1]
    channel = sample_aug_data_batch.shape[2]
    width = sample_aug_data_batch.shape[3]
    height = sample_aug_data_batch.shape[4]


    ori_patches = model_mae.patchify(images)
    sample_aug_patches = model_mae.patchify(sample_aug_data_batch.reshape([batch_size*aug_data_num,channel,width,height])) # [batch_size*aug_data_num,patch_num,patch_dim]
    aug_patches_num = aug_data_num * sample_aug_patches.shape[-2]
    patch_dim = sample_aug_patches.shape[-1]
    sample_aug_patches = sample_aug_patches.reshape([batch_size,aug_patches_num,patch_dim])

    ori_feats,_,_ = model_mae.forward_encoder(images,0)
    ori_feats_patches =  ori_feats[:, 1:, :]
    sample_aug_feats,_,_ = model_mae.forward_encoder(sample_aug_data_batch.reshape([batch_size*aug_data_num,channel,width,height]),0)
    sample_aug_feats_patches = sample_aug_feats[:, 1:, :] # [batch_size*aug_data_num,patch_num,patch_dim]
    sample_aug_feats_patches = sample_aug_feats_patches.reshape([batch_size,aug_patches_num,patch_dim]) # [batch_size,aug_patch_num,patch_dim]

    # pdb.set_trace()

    # Get the index of replaced patches considering representation space and raw image space
    # Rep Space
    masked_patches = ori_feats_patches[mask == 1].reshape([batch_size,-1,patch_dim])
    mask2aug_patches_dis_rep = torch.cdist(masked_patches,sample_aug_feats_patches,p=2)
    mask2aug_patches_dis_rep = mask2aug_patches_dis_rep / mask2aug_patches_dis_rep.sum()
    # Raw Space
    masked_patches = ori_patches[mask == 1].reshape([batch_size,-1,patch_dim])
    mask2aug_patches_dis_raw = torch.cdist(masked_patches,sample_aug_patches,p=2)
    mask2aug_patches_dis_raw = mask2aug_patches_dis_raw / mask2aug_patches_dis_raw.sum()

    aug_patches_idxs = torch.argsort(args_input.raw_rep_ratio*mask2aug_patches_dis_raw + (1-args_input.raw_rep_ratio)*mask2aug_patches_dis_rep)[:,:,0] # [batch_size, masked_patch_num]
   
    # replace patches for the original data
    images_patchify = model_mae.patchify(images)
    aug_images_patchify = model_mae.patchify(sample_aug_data_batch.reshape([batch_size*aug_data_num,sample_aug_data_batch.shape[-3],sample_aug_data_batch.shape[-2],sample_aug_data_batch.shape[-1]]))
    aug_images_patchify = aug_images_patchify.reshape([batch_size,aug_patches_num,patch_dim])
    # images_patchify[mask == 1] = torch.index_select(aug_images_patchify.reshape([batch_size*aug_patches_num,patch_dim]),dim=0,index=aug_patches_idxs.ravel())
    images_patchify[mask == 1] = torch.gather(aug_images_patchify, 1, aug_patches_idxs.unsqueeze(-1).expand(-1, -1, aug_images_patchify.shape[-1])).reshape([-1,patch_dim])
    images_aug = model_mae.unpatchify(images_patchify)

    # Vis the augment data
    # images_mask = model_mae.unpatchify(images_patchify*(1-mask).unsqueeze(-1))
    # idx_select = [3,4,6,8]
    # ts.save(images[idx_select],'/scratch/zx1673/codes/qlresearch/active_learning/deepALplus_xue/vis_case.png',ncols=5)
    # ts.save(images_mask[idx_select],'/scratch/zx1673/codes/qlresearch/active_learning/deepALplus_xue/vis_case_mask.png',ncols=5)
    # ts.save(images_aug[idx_select],'/scratch/zx1673/codes/qlresearch/active_learning/deepALplus_xue/vis_case_aug.png',ncols=5)
    # for j in range(batch_size):
    #     ts.save(sample_aug_data_batch[j],'/scratch/zx1673/codes/qlresearch/active_learning/deepALplus_xue/vis_case_sample_'+str(j)+'.png',ncols=5)
    # pdb.set_trace()

    return images_aug

def ColorPatchAug(model_mae, mask, images, args_task):
    imgs_patch = model_mae.patchify(images) # [10, 196, 768]
    # find aug patches
    aug_patch = imgs_patch[mask == 1] # [980, 768]
    patch_size = model_mae.patch_embed.patch_size[0]
    
    # aug aug_patches
    aug_patch = aug_patch.reshape(shape=(aug_patch.shape[0], patch_size, patch_size, 3)) # [980, 16, 16, 3]
    aug_patch = torch.einsum('npqc->ncpq', aug_patch) # [980, 3, 16, 16]
    color_jitter = transforms.Compose([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),transforms.ToTensor()])
    aug_patch = torch.stack([color_jitter(transforms.ToPILImage()(x)) for x in aug_patch])
    aug_patch = torch.einsum('ncpq->npqc', aug_patch)
    aug_patch = aug_patch.reshape(shape=(aug_patch.shape[0], patch_size*patch_size*3))
    imgs_patch[mask == 1] = aug_patch.to(imgs_patch.device)
    images_aug = model_mae.unpatchify(imgs_patch)
    norm = transforms.Compose([transforms.Normalize(mean = args_task['mean'], std = args_task['std'])])
    images_aug = torch.stack([norm(x) for x in images_aug])
    return images_aug

def FlipPatchAug(model_mae, mask, images, args_task):
    imgs_patch = model_mae.patchify(images) # [10, 196, 768]
    # find aug patches
    aug_patch = imgs_patch[mask == 1] # [980, 768]
    patch_size = model_mae.patch_embed.patch_size[0]
    
    # aug aug_patches
    aug_patch = aug_patch.reshape(shape=(aug_patch.shape[0], patch_size, patch_size, 3)) # [980, 16, 16, 3]
    aug_patch = torch.einsum('npqc->ncpq', aug_patch) # [980, 3, 16, 16]
    flip = transforms.Compose([transforms.RandomHorizontalFlip(p=1),transforms.ToTensor()])
    aug_patch = torch.stack([flip(transforms.ToPILImage()(x)) for x in aug_patch])
    aug_patch = torch.einsum('ncpq->npqc', aug_patch)
    aug_patch = aug_patch.reshape(shape=(aug_patch.shape[0], patch_size*patch_size*3))
    imgs_patch[mask == 1] = aug_patch.to(imgs_patch.device)
    images_aug = model_mae.unpatchify(imgs_patch)
    norm = transforms.Compose([transforms.Normalize(mean = args_task['mean'], std = args_task['std'])])
    images_aug = torch.stack([norm(x) for x in images_aug])
    return images_aug

def MaskPatchAug(model_mae, mask, images, args_task):
    imgs_patch = model_mae.patchify(images) # [10, 196, 768]
    imgs_patch[mask == 1] = 0

    images_aug = model_mae.unpatchify(imgs_patch)
    norm = transforms.Compose([transforms.Normalize(mean = args_task['mean'], std = args_task['std'])])
    images_aug = torch.stack([norm(x) for x in images_aug])
    return images_aug

def GaussianPatchAug(model_mae, mask, images, args_task):
    imgs_patch = model_mae.patchify(images) # [10, 196, 768]
    # find aug patches
    aug_patch = imgs_patch[mask == 1] # [980, 768]
    aug_patch = 0.1 * torch.randn(aug_patch.size()).to(imgs_patch.device)
    imgs_patch[mask == 1] = aug_patch.to(imgs_patch.device)
    images_aug = model_mae.unpatchify(imgs_patch)
    norm = transforms.Compose([transforms.Normalize(mean = args_task['mean'], std = args_task['std'])])
    images_aug = torch.stack([norm(x) for x in images_aug])
    return images_aug

def AutoPatchAug(model_mae, mask, images, args_task):
    imgs_patch = model_mae.patchify(images) # [10, 196, 768]
    # find aug patches
    aug_patch = imgs_patch[mask == 1] # [980, 768]
    patch_size = model_mae.patch_embed.patch_size[0]
    
    # aug aug_patches
    aug_patch = aug_patch.reshape(shape=(aug_patch.shape[0], patch_size, patch_size, 3)) # [980, 16, 16, 3]
    aug_patch = torch.einsum('npqc->ncpq', aug_patch) # [980, 3, 16, 16]
    flip = transforms.Compose([transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),transforms.ToTensor()])
    aug_patch = torch.stack([flip(transforms.ToPILImage()(x)) for x in aug_patch])
    aug_patch = torch.einsum('ncpq->npqc', aug_patch)
    aug_patch = aug_patch.reshape(shape=(aug_patch.shape[0], patch_size*patch_size*3))
    imgs_patch[mask == 1] = aug_patch.to(imgs_patch.device)
    images_aug = model_mae.unpatchify(imgs_patch)
    norm = transforms.Compose([transforms.Normalize(mean = args_task['mean'], std = args_task['std'])])
    images_aug = torch.stack([norm(x) for x in images_aug])
    return images_aug

def RandPatchAug(model_mae, mask, images, args_task):
    imgs_patch = model_mae.patchify(images) # [10, 196, 768]
    # find aug patches
    aug_patch = imgs_patch[mask == 1] # [980, 768]
    patch_size = model_mae.patch_embed.patch_size[0]
    
    # aug aug_patches
    aug_patch = aug_patch.reshape(shape=(aug_patch.shape[0], patch_size, patch_size, 3)) # [980, 16, 16, 3]
    aug_patch = torch.einsum('npqc->ncpq', aug_patch) # [980, 3, 16, 16]transforms.RandAugment(num_ops=5)
    flip = transforms.Compose([transforms.RandAugment(num_ops=5),transforms.ToTensor()])
    aug_patch = torch.stack([flip(transforms.ToPILImage()(x)) for x in aug_patch])
    aug_patch = torch.einsum('ncpq->npqc', aug_patch)
    aug_patch = aug_patch.reshape(shape=(aug_patch.shape[0], patch_size*patch_size*3))
    imgs_patch[mask == 1] = aug_patch.to(imgs_patch.device)
    images_aug = model_mae.unpatchify(imgs_patch)
    norm = transforms.Compose([transforms.Normalize(mean = args_task['mean'], std = args_task['std'])])
    images_aug = torch.stack([norm(x) for x in images_aug])
    return images_aug

def Local_Aug(args_input,args_task,images,labels,model_mae,adapter,sample_aug_data_batch):
    local_strategy = args_input.local_strategy
    aug_strategy = args_input.aug_strategy

    feats,_,_ = model_mae.forward_encoder(images,0)
    patches = feats[:, 1:, :]
    if args_input.use_global_avg == False:
        cls = feats[:, 0].unsqueeze(1)
    else:
        cls = feats[:, 1:, :].mean(dim=1).unsqueeze(1)
    
    if local_strategy == 'Att':
        cls_ext = cls.repeat([1,patches.shape[1],1])
        att = 1 - torch.cosine_similarity(patches,cls_ext,dim=-1) # [batch_size, patch_num]
        # top_k = int(att.shape[-1]*top_ratio)
        feats,mask,ids_restore = model_mae.forward_encoder(images,args_input.local_ratio,certain_mask=True,att=att)
    
    elif local_strategy == 'Att_Last':
        cls = feats[:, 1:, :].mean(dim=1).unsqueeze(1)
        att = model_mae.forward_att_map(images)
        att = 1 - att
        feats,mask,ids_restore = model_mae.forward_encoder(images,args_input.local_ratio,certain_mask=True,att=att)
        attention_maps = [block for block in model_mae.blocks]
    elif local_strategy == 'Rand':
        # pdb.set_trace()
        feats,_,_ = model_mae.forward_encoder(images,0)
        N =  patches.shape[0]  # batch
        L = patches.shape[1] # Length
        att = torch.rand(N, L, device=images.device)  # noise in [0, 1]
        feats,mask,ids_restore = model_mae.forward_encoder(images,args_input.local_ratio,certain_mask=True,att=att)

    elif local_strategy == 'Saliency':
        com_model = CombineModel(model_mae,adapter,args_input)
        saliency = Saliency(com_model)
        att = saliency.attribute(images, target=labels.squeeze())
        att = 1 - model_mae.patchify(att).mean(-1)
        feats,mask,ids_restore = model_mae.forward_encoder(images,args_input.local_ratio,certain_mask=True,att=att)

    elif local_strategy == 'DeepLift':
        com_model = CombineModel(model_mae,adapter,args_input)
        dl = DeepLift(com_model)
        att = dl.attribute(images, target=labels.squeeze())
        att = 1 - model_mae.patchify(att).mean(-1)
        feats,mask,ids_restore = model_mae.forward_encoder(images,args_input.local_ratio,certain_mask=True,att=att)
    else:
        raise ValueError('No Local Strategy named '+str(local_strategy))
    
    # mask: 0 is keep, 1 is to augment
    mask_score = ((1-mask)*att).sum(-1) / att.sum(-1)
    labels_aug = F.one_hot(labels,num_classes=args_task['num_class'])
    
    if aug_strategy == 'Self':
        imgs_aug = SelfAug(images,feats,model_mae,ids_restore,mask)
        if args_input.label_smooth_local:
            labels_aug = Label_Smooth(labels_aug,mask_score)
    elif aug_strategy == 'SamplePatchAug':
        imgs_aug = SamplePatchAug(model_mae,args_input,sample_aug_data_batch, mask, images)
        if args_input.label_smooth_local:
            labels_aug = Label_Smooth(labels_aug,mask_score)
    elif aug_strategy == 'ColorPatchAug':
        imgs_aug = ColorPatchAug(model_mae, mask, images, args_task)
        if args_input.label_smooth_local:
            labels_aug = Label_Smooth(labels_aug,mask_score)
    elif aug_strategy == 'FlipPatchAug':
        imgs_aug = FlipPatchAug(model_mae, mask, images, args_task)
        if args_input.label_smooth_local:
            labels_aug = Label_Smooth(labels_aug,mask_score)
    elif aug_strategy == 'MaskPatchAug':
        imgs_aug = MaskPatchAug(model_mae, mask, images, args_task)
        if args_input.label_smooth_local:
            labels_aug = Label_Smooth(labels_aug,mask_score)
    elif aug_strategy == 'GaussianPatchAug':
        imgs_aug = GaussianPatchAug(model_mae, mask, images, args_task)
        if args_input.label_smooth_local:
            labels_aug = Label_Smooth(labels_aug,mask_score)
    elif aug_strategy == 'AutoPatchAug':
        imgs_aug = AutoPatchAug(model_mae, mask, images, args_task)
        if args_input.label_smooth_local:
            labels_aug = Label_Smooth(labels_aug,mask_score)
    elif aug_strategy == 'RandPatchAug':
        imgs_aug = RandPatchAug(model_mae, mask, images, args_task)
        if args_input.label_smooth_local:
            labels_aug = Label_Smooth(labels_aug,mask_score)
    else:
        raise ValueError('No such a aug strategy named ' + str(aug_strategy))
    
    
    # Codes to store the cases
    # idx_select = [3,4,6,8]
    # if args_input.draw_vis:
    #     images_patchify = model_mae.patchify(images)
    #     images_mask = model_mae.unpatchify(images_patchify*(1-mask).unsqueeze(-1))
    #     vis_path = '/scratch/zx1673/codes/qlresearch/active_learning/deepALplus_xue/vis_figs/' + str(aug_strategy) + '_' + str(local_strategy) + '_' + str(args_input.local_ratio)
    #     if os.path.exists(vis_path) is False:
    #         os.makedirs(vis_path)
    #     ts.save(images_mask,os.path.join(vis_path,'vis_case_mask.png'),ncols=5)
    #     ts.save(images,os.path.join(vis_path,'vis_case.png'),ncols=5)
    #     ts.save(imgs_aug,os.path.join(vis_path,'vis_case_aug.png'),ncols=5)
    #     pdb.set_trace()
    return imgs_aug,labels_aug












def vis_att_mask(args,images,topk_idx,att,top_ratio=0.5):
    num_show = images.shape[0]
    patch_num = int(att.shape[1]**0.5)
    att = torch.ones_like(att)
    for b in range(num_show):
        att[b,topk_idx[b]] = 0
    att = att.unsqueeze(1).reshape([num_show,patch_num,patch_num])
    # att = att.repeat([1,patch_size,patch_size])
    att = F.interpolate(att.unsqueeze(1),size=[args.input_size,args.input_size])
    att = att * images
    ts.save(att, os.path.join(save_path,'att_'+str(top_ratio)+'.jpg'))


def vis_att_mask(dataset,model,num_show,m_ratio,device,save_path):
    
    images = torch.concat([dataset[i][0].unsqueeze(0) for i in range(num_show)],dim=0)
    labels = [dataset[i][1] for i in range(num_show)]
    images = images.to(device)
    feats,_,_ = model.forward_encoder(images,0)

    ts.save(images, os.path.join(save_path,'imgs.jpg'))
    
    patches = feats[:, 1:, :]
    cls = feats[:, 0].unsqueeze(1)

    cls_ext = cls.repeat([1,patches.shape[1],1])

    # query
    # att = 1 - (cls_ext*patches).mean(-1)
    att = 1 - torch.cosine_similarity(patches,cls_ext,dim=-1)
    patch_num = int(att.shape[1]**0.5)
    patch_size = int(args.input_size/patch_num)
    top_k = int(patch_num*patch_num*m_ratio)  
    # select
    _, topk_idx = torch.topk(att,top_k,dim=-1)
    topk_idx = topk_idx.cpu().numpy().tolist()
    
    att = torch.ones_like(att)
    for b in range(num_show):
        att[b,topk_idx[b]] = 0
    
    att = att.unsqueeze(1).reshape([num_show,patch_num,patch_num])
    # att = att.repeat([1,patch_size,patch_size])
    att = F.interpolate(att.unsqueeze(1),size=[args.input_size,args.input_size])
    att = att * images
    ts.save(att, os.path.join(save_path,'test_att_'+str(m_ratio)+'.jpg'))






"""
Localization & Augmentation Method
"""
# def Local_Aug(args,model,model_lrp,adapter,images,labels):
#     local_strategy = args.local_strategy
#     aug_strategy = args.aug_strategy
#     if local_strategy == 'Att':
#         feats,_,_ = model.forward_encoder(images,0)
#         att,topk_idx = adapter.forward_att(feats)
#         feats,mask,ids_restore = model.forward_encoder(images,args.local_ratio,certain_mask=True,att=att)
    
#     elif local_strategy == 'CE':
#         feats,_,_ = model.forward_encoder(images,0)
#         att = adapter.forward_cls(feats,labels)
#         feats,mask,ids_restore = model.forward_encoder(images,args.local_ratio,certain_mask=True,att=att)
    
#     elif local_strategy == 'EP':
#         feats,_,_ = model.forward_encoder(images,0)
#         patches = feats[:, 1:, :]
#         if args.use_global_avg == False:
#             cls = feats[:, 0].unsqueeze(1)
#         else:
#             cls = feats[:, 1:, :].mean(dim=1).unsqueeze(1)
#         com_model = CombineModel(model,adapter,args)
#         att = 1 - extremal_perturbation_batch(com_model,images,labels,reward_func=contrastive_reward,areas=[1-args.local_ratio],max_iter=200)
#         # att -> [B,P**2]  
#         feats,mask,ids_restore = model.forward_encoder(images,args.local_ratio,certain_mask=True,att=att)
#     elif local_strategy == 'GradCAM':
#         feats,_,_ = model.forward_encoder(images,0)
#         patches = feats[:, 1:, :]
#         if args.use_global_avg == False:
#             cls = feats[:, 0].unsqueeze(1)
#         else:
#             cls = feats[:, 1:, :].mean(dim=1).unsqueeze(1)
#         com_model = CombineModel(model,adapter,args)
#         guided_gc = GuidedGradCam(com_model,com_model.model.blocks[-1])
#         att =  guided_gc.attribute(images, labels.squeeze())
#         att = model.patchify(att).mean(-1)
#         feats,mask,ids_restore = model.forward_encoder(images,args.local_ratio,certain_mask=True,att=att)

#     elif local_strategy == 'InteGrad':
#         """
#         Under Construction, Too Large
#         """
#         feats,_,_ = model.forward_encoder(images,0)
#         patches = feats[:, 1:, :]
#         if args.use_global_avg == False:
#             cls = feats[:, 0].unsqueeze(1)
#         else:
#             cls = feats[:, 1:, :].mean(dim=1).unsqueeze(1)
#         com_model = CombineModel(model,adapter,args)
#         ig = IntegratedGradients(model)
#         baseline = torch.zeros_like(images)
#         pdb.set_trace()
#         attributions, delta = ig.attribute(images, baseline, target=labels.squeeze(), return_convergence_delta=True)
#         pdb.set_trace()
#     elif local_strategy == 'Saliency':
#         feats,_,_ = model.forward_encoder(images,0)
#         patches = feats[:, 1:, :]
#         if args.use_global_avg == False:
#             cls = feats[:, 0].unsqueeze(1)
#         else:
#             cls = feats[:, 1:, :].mean(dim=1).unsqueeze(1)
#         com_model = CombineModel(model,adapter,args)
#         saliency = Saliency(com_model)
#         att = saliency.attribute(images, target=labels.squeeze())
#         att = 1 - model.patchify(att).mean(-1)
#         feats,mask,ids_restore = model.forward_encoder(images,args.local_ratio,certain_mask=True,att=att)
#     elif local_strategy == 'Occlusion':
#         feats,_,_ = model.forward_encoder(images,0)
#         patches = feats[:, 1:, :]
#         if args.use_global_avg == False:
#             cls = feats[:, 0].unsqueeze(1)
#         else:
#             cls = feats[:, 1:, :].mean(dim=1).unsqueeze(1)
#         com_model = CombineModel(model,adapter,args)
#         ablator = Occlusion(com_model)
#         att = ablator.attribute(images, target=labels.squeeze(), sliding_window_shapes=(3,model.patch_embed.patch_size[0]*4,model.patch_embed.patch_size[0]*4))
#     elif local_strategy == 'DeepLift':
#         feats,_,_ = model.forward_encoder(images,0)
#         patches = feats[:, 1:, :]
#         if args.use_global_avg == False:
#             cls = feats[:, 0].unsqueeze(1)
#         else:
#             cls = feats[:, 1:, :].mean(dim=1).unsqueeze(1)
#         com_model = CombineModel(model,adapter,args)
#         dl = DeepLift(com_model)
#         att = dl.attribute(images, target=labels.squeeze())
#         att = 1 - model.patchify(att).mean(-1)
#         feats,mask,ids_restore = model.forward_encoder(images,args.local_ratio,certain_mask=True,att=att)
#     elif local_strategy == 'Transformer_Attribution':
#         attr_lst = []
#         # attr_generator = models_mae_lrp.LRP_MAE(model_lrp,adapter)
#         model_lrp.load_state_dict(copy.deepcopy(model.state_dict()))
#         for idx in range(images.shape[0]):
#             model_lrp.zero_grad()
#             model_lrp = model_lrp.to(args.device)
#             attr = models_mae_lrp.LRP_MAE(model_lrp,adapter,images[idx].unsqueeze(0), method="transformer_attribution", index=labels[idx].cpu().data.numpy(),device=args.device).detach()
#             attr_lst.append(attr)
        
#         model_lrp.release()
#         att = 1 - torch.concat(attr_lst)
#         feats,mask,ids_restore = model.forward_encoder(images,args.local_ratio,certain_mask=True,att=att)
    
#     elif local_strategy == 'Last_Att':
#         attr_lst = []
#         # attr_generator = models_mae_lrp.LRP_MAE(model_lrp,adapter)
#         model_lrp.load_state_dict(copy.deepcopy(model.state_dict()))
#         for idx in range(images.shape[0]):
#             model_lrp.zero_grad()
#             model_lrp = model_lrp.to(args.device)
#             attr = models_mae_lrp.LRP_MAE(model_lrp,adapter,images[idx].unsqueeze(0), method='last_layer_attn', index=labels[idx].cpu().data.numpy(),device=args.device).detach()
#             attr_lst.append(attr.unsqueeze(0))
        
#         model_lrp.release()
#         att = 1 - torch.concat(attr_lst)
#         feats,mask,ids_restore = model.forward_encoder(images,args.local_ratio,certain_mask=True,att=att)
#     else:
#         att = None
#         mask = None
    

#     if aug_strategy == 'Self':
#         imgs_aug = SelfAug(images,feats,model,ids_restore,mask)
#     elif aug_strategy == 'Rand':
#         imgs_aug = RandAug(images)
#     elif aug_strategy == 'Normal':
#         imgs_aug = NormalAug(images)
#     elif aug_strategy == 'MixUp':
#         imgs_aug,labels = MixUp(args,images,labels)
#     elif aug_strategy == 'CutMix':
#         imgs_aug,labels = CutMix(args,images,labels)
#     elif aug_strategy == 'CutMixUp':
#         imgs_aug,labels = CutMixUp(args,images,labels)
#     elif aug_strategy == 'ManifoldMixUp':
#         feats_aug,labels = ManifoldMixUp(args,images,labels,model)
#         imgs_aug = images
#     elif aug_strategy == 'TokenMixUp':
#         pass
    
#     if aug_strategy != 'ManifoldMixUp':
#         feats_aug,_,_ = model.forward_encoder(imgs_aug,0)
#         if args.use_global_avg:
#             feats_aug = feats_aug[:, 1:, :].mean(dim=1)
#         else:
#             feats_aug = feats_aug[:, 0]

#     # vis_images_aug = imgs_aug[:6]
#     # ts.save(vis_images_aug,'vis_'+aug_strategy+'.png')
#     # vis_images = images[:6]
#     # ts.save(vis_images,'vis_img.png')
#     # model.eval()
#     # feats_aug,_,_ = model.forward_encoder(imgs_aug,0)
#     return feats_aug, imgs_aug, labels
    
"""
Vis codes
"""
# # feats,mask,ids_restore = model.forward_encoder(images,args.local_ratio)
# pred = model.forward_decoder(feats, ids_restore)                
# mask = mask.detach()
# mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
# mask = model.unpatchify(mask)
# img_mask = (1-mask)*images
# # loss = self.forward_loss(imgs, pred, mask)
# mask = model.unpatchify(mask)
# pred = model.unpatchify(pred)
# imgs_aug = images * (1 - mask) + pred * mask
    

