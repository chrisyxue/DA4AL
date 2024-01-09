import warnings
import argparse
import sys
import os
import re

import yaml
from ast import literal_eval
import copy

DATASETS = ['CIFAR10','CIFAR100','CIFAR','TinyImageNet','DermaMNIST','BreastMNIST', 'PathMNIST','PneumoniaMNIST', 'ChestMNIST','TissueMNIST','OCTMNIST','OrganAMNIST','OrganCMNIST','OrganSMNIST','BloodMNIST']
MODELS = ['resnet18','resnet34','resnet50','vgg16','alexnet','mobilenet','resnet18_pretrain','resnet34_pretrain','resnet50_pretrain','resnet18_pretrain_ft','resnet34_pretrain_ft','resnet50_pretrain_ft']
STRATEGY = ['EntropySampling','LeastConfidence','MarginSampling','LeastConfidenceDropout','MarginSamplingDropout','EntropySamplingDropout','KCenterGreedy','RandomSampling','MeanSTD','BALDDropout','BadgeSampling']
DA_STRATEGY = ['No','Self','Sample','SamplePatchAug','ColorPatchAug','MaskPatchAug','FlipPatchAug','AutoPatchAug','RandPatchAug','GaussianPatchAug','PCutMix','Normal','Auto','Rand','MixUp','CutMix','CutMixUp','ManifoldMixUp']
SEMI_LOSS = ['EntMin','L2Cons','L2Relation','L2RelationNorm']
MAE_PRETRAIN_DATA = ['General','Source','Source_FT','MoCo','DINO','CXR']
LOCAL_STRATEGY = ['Rand','Att','Saliency','DeepLift','Att_Last']


# Change to your own path
WEIGHT_PATH_BASE = {
    'MoCo': '/home/zhiyu/models/vit/mocov3-vit-b-300ep.pth',
    'DINO': '/home/zhiyu/models/vit/dino_vitbase16_pretrain.pth',
	'CXR': '/home/zhiyu/models/vit/vit-b_CXR_0.5M_mae.pth'
}


def get_args():
	parser = argparse.ArgumentParser(description='Extended Deep Active Learning Toolkit')
	#basic arguments
	parser.add_argument('--ALstrategy', '-a', default='EntropySampling', type=str, help='name of active learning strategies')
	# parser.add_argument('--quota', '-q', default=1000, type=int, help='quota of active learning')
	parser.add_argument('--round', '-r', default=100, type=int, help='number of query rounds')
	parser.add_argument('--batch', '-b', default=128, type=int, help='batch size in one active learning iteration')
	parser.add_argument('--iteration', '-t', default=5, type=int, help='time of repeat the experiment')
	parser.add_argument('--data_path', type=str, default='./../data', help='Path to where the data is')
	parser.add_argument('--log_name', type=str, default='test.log', help='middle outputs')
	#parser.add_argument('--help', '-h', default=False, action='store_true', help='verbose')
	parser.add_argument('--cuda', action='store_true', help='If training is to be done on a GPU')
	#parser.add_argument('--model', '-m', default='ResNet18', type=str, help='model name')
	parser.add_argument('--initseed', '-s', default = 1000, type = int, help = 'Initial pool of labeled data')
	parser.add_argument('--gpu', '-g', default = 0, type = str, help = 'which gpu')
	parser.add_argument('--seed', default=4666, type=int, help='random seed')
	
	# lpl
	parser.add_argument('--lpl_epoches', type=int, default=20, help='lpl epoch num after detach')
	# ceal
	parser.add_argument('--delta', type=float, default=5 * 1e-5, help='value of delta in ceal sampling')
	#hyper parameters
	parser.add_argument('--train_epochs', type=int, default=100, help='Number of training epochs')

	#specific parameters
	parser.add_argument('--latent_dim', type=int, default=32, help='The dimensionality of the VAE latent dimension')

	parser.add_argument('--beta', type=float, default=1, help='Hyperparameter for training. The parameter for VAE')
	parser.add_argument('--num_adv_steps', type=int, default=1, help='Number of adversary steps taken for every task model step')
	parser.add_argument('--num_vae_steps', type=int, default=2, help='Number of VAE steps taken for every task model step')
	parser.add_argument('--adversary_param', type=float, default=1, help='Hyperparameter for training. lambda2 in the paper')

    # xuezhiyu
	parser.add_argument('--pretrained_model_path', type=str, default='./basicmodel', help='Path to where the pretrained model is')
	parser.add_argument('--dataset_name', '-d', default='CIFAR10', choices=DATASETS, type=str, help='dataset name')
	parser.add_argument('--model', '-m', default='resnet18', type=str, choices=MODELS, help='model name')
	parser.add_argument('--out_path', type=str, default='./../results', help='Path to where the output log will be')
	parser.add_argument('--draw_vis', action='store_true', help='If training is to be done on a GPU')
	args = parser.parse_args()
	return args



def get_args_finetune():
	parser = argparse.ArgumentParser(description='Extended Deep Active Learning Toolkit')
	#basic arguments
	parser.add_argument('--ALstrategy', '-a', default='EntropySampling', type=str, help='name of active learning strategies')
	# parser.add_argument('--quota', '-q', default=1000, type=int, help='quota of active learning')
	parser.add_argument('--round', '-r', default=100, type=int, help='number of query rounds')
	parser.add_argument('--batch', '-b', default=128, type=int, help='batch size in one active learning iteration')
	parser.add_argument('--iteration', '-t', default=5, type=int, help='time of repeat the experiment')
	parser.add_argument('--data_path', type=str, default='./../data', help='Path to where the data is')
	parser.add_argument('--log_name', type=str, default='test.log', help='middle outputs')
	#parser.add_argument('--help', '-h', default=False, action='store_true', help='verbose')
	parser.add_argument('--cuda', action='store_true', help='If training is to be done on a GPU')
	#parser.add_argument('--model', '-m', default='ResNet18', type=str, help='model name')
	parser.add_argument('--initseed', '-s', default = 1000, type = int, help = 'Initial pool of labeled data')
	parser.add_argument('--gpu', '-g', default = 0, type = str, help = 'which gpu')
	parser.add_argument('--seed', default=4666, type=int, help='random seed')
	
	# lpl
	parser.add_argument('--lpl_epoches', type=int, default=20, help='lpl epoch num after detach')
	# ceal
	parser.add_argument('--delta', type=float, default=5 * 1e-5, help='value of delta in ceal sampling')
	#hyper parameters
	parser.add_argument('--train_epochs', type=int, default=100, help='Number of training epochs')

	#specific parameters
	parser.add_argument('--latent_dim', type=int, default=32, help='The dimensionality of the VAE latent dimension')

	parser.add_argument('--beta', type=float, default=1, help='Hyperparameter for training. The parameter for VAE')
	parser.add_argument('--num_adv_steps', type=int, default=1, help='Number of adversary steps taken for every task model step')
	parser.add_argument('--num_vae_steps', type=int, default=2, help='Number of VAE steps taken for every task model step')
	parser.add_argument('--adversary_param', type=float, default=1, help='Hyperparameter for training. lambda2 in the paper')

    # xuezhiyu
	parser.add_argument('--pretrained_model_path', type=str, default='./basicmodel', help='Path to where the pretrained model is')
	parser.add_argument('--dataset_name', '-d', default='CIFAR10', choices=DATASETS, type=str, help='dataset name')
	# parser.add_argument('--model', '-m', default='resnet18', type=str, choices=MODELS, help='model name')
	parser.add_argument('--out_path', type=str, default='./../results', help='Path to where the output log will be')

	# For Finetune MAE
	parser.add_argument('--model_mae', type=str, default='mae_vit_base_patch16', help='Model Structure of MAE')
	parser.add_argument('--pretrained_data', type=str, default='General', choices=MAE_PRETRAIN_DATA, help='The data for pretrained model')
	parser.add_argument('--adapter',default='MLP',choices=['MLP','ResMLP'],type=str,help='structure of adapter')
	parser.add_argument('--norm_pix_loss', action='store_true',help='Use (per-patch) normalized pixels as targets for computing loss')
	parser.add_argument('--checkpoint_source',default='/model/model.pkl',type=str,help='The Model Path for MAE Model Pretrained on Source Data')
	parser.add_argument('--checkpoint_sourceft',default='/model/model.pkl',type=str,help='The Model Path for General MAE Model Pretrained on Source Data')
	parser.add_argument('--use_global_avg', action='store_true',help='Use avg representation or cls')
	
	# For SemiSup
	parser.add_argument('--tau', default=0.0, type=float, help='The factor for the unl loss')
	parser.add_argument('--num_unl', default=16, type=float, help='The number of unl data per batch')
	parser.add_argument('--unl_sample',default='Random',type=str,choices=['Random'])
	parser.add_argument('--semi_loss',default='L2Cons',type=str,choices=SEMI_LOSS)
	parser.add_argument('--semi_decay',default=1,type=float)
	

	parser.add_argument('--aug_strategy',default='No',type=str,choices=DA_STRATEGY,help='strategy for label irrelevant patches augmentation')
	parser.add_argument('--local_strategy',default='Rand',type=str,choices=LOCAL_STRATEGY,help='strategy for local irrelevant patches')
	parser.add_argument('--local_ratio',default=0.75,type=float,help='ratio for masking local irrelevant patches')
	parser.add_argument('--label_smooth_local', action='store_true',help='use label smooth for local strategy')

	# For sample patches augmentation
	parser.add_argument('--dis_matrix_path',default='',type=str,help='path of distance matrix')
	parser.add_argument('--sample_top_k',default=5,type=int,help='sample top k samples for patch aug or as augmented data')
	parser.add_argument('--raw_rep_ratio',default=0.5,type=float,help='the trade-off ratio for sample from raw space and rep space for SamplePatchAug')
	parser.add_argument('--draw_vis', action='store_true', help='If training is to be done on a GPU')
	args = parser.parse_args()
	return args

