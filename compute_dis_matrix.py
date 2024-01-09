"""
Compute the Distance Matrix for Samples
"""

import numpy as np
import torch
from utils import get_dataset
from pprint import pprint

torch.set_printoptions(profile='full')
import os
import random

import arguments
from parameters import *
from utils import *
import pandas as pd
from torch.utils.data import DataLoader
from analysis.rep_utils import *

def mae_encode(model_mae,x,args):
	with torch.no_grad():
		feats,_,_ = model_mae.forward_encoder(x, mask_ratio=0)
		if args.use_global_avg:
			feats = feats[:, 1:, :].mean(dim=1)
		else:
			feats = feats[:, 0]	
	return feats

def dis_matrix(train_dataloader,mae_model):
	feats_all = []
	y_all = []
	x_all = []
	
	for batch_idx, (x, y, idxs) in enumerate(train_dataloader):
		print('batch:'+str(batch_idx)+' of '+str(len(train_dataloader)))
		x,y = x.to(device), y.to(device)
		feats = mae_encode(mae_model,x,args_input)
		feats_all.append(feats.detach().cpu())
		y_all.append(y.detach().cpu())
		dim = feats.shape[-1]
	
	feats_all = torch.concat(feats_all)
	y_all = torch.concat(y_all)
    
	dis_matrix = []
	num_sample = feats_all.shape[0]
	for i in range(feats_all.shape[0]):
		dis = torch.norm((feats_all[i].unsqueeze(0).repeat([num_sample,1])-feats_all),dim=-1,p=2).unsqueeze(0)
		dis_matrix.append(dis)
	
	dis_matrix = torch.concat(dis_matrix)

	return dis_matrix, y_all

def fix_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.enabled  = True
	torch.backends.cudnn.benchmark= True

def mean_var_aubc(all_acc):
	acc_m = []
	for i in range(len(all_acc)):
		acc_m.append(get_aubc(args_input.quota, NUM_QUERY, all_acc[i]))
	mean_acc,stddev_acc = get_mean_stddev(acc_m)
	return mean_acc, stddev_acc

def mean_var(all_acc):
	mean_acc = np.array(all_acc).mean(0)
	std_acc = np.array(all_acc).std(0)
	return mean_acc, std_acc

# parameters
args_input = arguments.get_args_finetune()
NUM_QUERY = args_input.batch
NUM_INIT_LB = args_input.initseed
NUM_ROUND = args_input.round
DATA_NAME = args_input.dataset_name
STRATEGY_NAME = args_input.ALstrategy
args_input.quota = NUM_ROUND * NUM_QUERY
SEED = args_input.seed

# args_input.out_path = os.path.join(args_input.out_path,'dis_matrix')
os.environ['TORCH_HOME'] = args_input.pretrained_model_path
os.environ["CUDA_VISIBLE_DEVICES"] = str(args_input.gpu)

# fix random seed
fix_seed(SEED)

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# device = 'cuda'
# data, network, strategy
args_task = args_pool[DATA_NAME]
dataset = get_dataset(args_input.dataset_name, args_task, args_input)				# load dataset




"""
load data
"""
dataset.handler(dataset.X_train, dataset.Y_train, dataset.args_task['transform_train'])
train_dataset = dataset.handler(dataset.X_train, dataset.Y_train, dataset.args_task['transform_train'])
train_dataloader = DataLoader(train_dataset,batch_size=256,shuffle=False,drop_last=False)



"""
load Model
"""
# args_input.pretrained_data = 'General'
net_general = get_net_mae(args_input.dataset_name, args_task, device, args_input)
mae_general = net_general.model_mae
mae_general.eval()
mae_general.to(device)
save_dir = render_res_file_mae_dis_matrix(args_input)

dis_mat, y_all = dis_matrix(train_dataloader,mae_model=mae_general)
assert((y_all == dataset.Y_train).sum().item() == dis_mat.shape[0])
_, indices = torch.sort(dis_mat,dim=-1)

save_dir = render_res_file_mae_dis_matrix(args_input)
if os.path.exists(save_dir) is False:
	os.makedirs(save_dir)
torch.save(dis_mat,os.path.join(save_dir,'dis_mat.pt'))
torch.save(indices,os.path.join(save_dir,'dis_indices.pt'))







