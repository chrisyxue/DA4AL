"""
Vis study for MAE

Zhiyu Xue
"""

import sys
import os
import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import pdb

# def show_image(image, title='',mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
def show_image(image, title='',mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]):
    # image is [H, W, 3]
    mean = np.array(mean)
    std = np.array(std)
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * std + mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def run_one_image(x, model, save_path=''):
    # x = torch.tensor(img)
    """
    Visualize the image, masked, and reconstruction
    x -> [c,h,w]
    """

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    # x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=0.75)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', x)
    x = x.detach().cpu()
    # import pdb;pdb.set_trace()
    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # Compute the rec loss for images
    rec_loss = (im_paste - x) ** 2
    rec_loss = rec_loss.mean().item()  # [N, L], mean loss per patch
    # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    # pdb.set_trace()
    
    # make the plt figure larger
    plt.figure()
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")
    
    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)

    # show_image(im_paste[0], "reconstruction + visible (rec loss="+str(round(rec_loss,4))+")" + "(train loss=" +str(round(loss.item(),4))+")")
    show_image(im_paste[0], "reconstruction + visible (rec loss="+str(round(rec_loss,4))+")")
    
    plt.savefig(save_path)

    return True

def run_batch_images(x, y, model, save_dir=''):
    """
    Visualize the batch, masked, and reconstruction
    x -> [n,c,h,w]
    """
    for idx in range(x.shape[0]):
        xi = x[idx]
        yi = y[idx].item()
        fig_name = str(idx) + '_label_' + str(yi)
        flag = run_one_image(xi, model, save_path=os.path.join(save_dir,fig_name))

    return flag
