from handlers import  CIFAR10_Handler, MedMNIST_ThreeChannel_Handler, MedMNIST_OneChannel_Handler

from datasets_lib.data import get_CIFAR10, get_PneumoniaMNIST, get_DermaMNIST, get_BreastMNIST, get_PathMNIST, \
							get_PneumoniaMNIST, get_ChestMNIST, get_TissueMNIST, get_OCTMNIST, get_OrganAMNIST, get_OrganCMNIST, get_OrganSMNIST, get_BloodMNIST

from nets import Net, CIFAR10_Net, MedMNIST_Net

from nets_mae import Net_MAE
from mae.models_adapter import Adapter

from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
								LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
								KCenterGreedy, BALDDropout,  \
								MeanSTD, BadgeSampling
from parameters import *
import sys
import os
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import torch

#Handler
def get_handler(name):
	if name == 'CIFAR10':
		return CIFAR10_Handler
	elif name == 'PneumoniaMNIST':
		return CIFAR10_Handler
	elif name == 'DermaMNIST':
		return MedMNIST_ThreeChannel_Handler
	elif name == 'BreastMNIST':
		return MedMNIST_OneChannel_Handler
	elif name == 'PathMNIST':
		return MedMNIST_ThreeChannel_Handler 
	elif name == 'PneumoniaMNIST':
		return MedMNIST_OneChannel_Handler 
	elif name == 'ChestMNIST':
		return MedMNIST_OneChannel_Handler	
	elif name == 'TissueMNIST':
		return MedMNIST_OneChannel_Handler
	elif name == 'OCTMNIST':
		return MedMNIST_OneChannel_Handler		
	elif name == 'OrganAMNIST':
		return MedMNIST_OneChannel_Handler
	elif name == 'OrganCMNIST':
		return MedMNIST_OneChannel_Handler
	elif name == 'OrganSMNIST':
		return MedMNIST_OneChannel_Handler
	elif name == 'BloodMNIST':
		return MedMNIST_ThreeChannel_Handler 
	else: 
		raise NotImplementedError

def get_dataset(name, args_task, args):
	if name == 'CIFAR10':
		return get_CIFAR10(get_handler(name), args_task, args)
	elif name == 'DermaMNIST':
		return get_DermaMNIST(get_handler(name), args_task, args)
	elif name == 'BreastMNIST':
		return get_BreastMNIST(get_handler(name), args_task, args)
	elif name == 'PathMNIST':
		return get_PathMNIST(get_handler(name), args_task, args)
	elif name == 'PneumoniaMNIST':
		return get_PneumoniaMNIST(get_handler(name), args_task, args)
	elif name == 'ChestMNIST':
		return get_ChestMNIST(get_handler(name), args_task, args)
	elif name == 'TissueMNIST':
		return get_TissueMNIST(get_handler(name), args_task, args)
	elif name == 'OCTMNIST':
		return get_OCTMNIST(get_handler(name), args_task, args)
	elif name == 'OrganAMNIST':
		return get_OrganAMNIST(get_handler(name), args_task, args)
	elif name == 'OrganCMNIST':
		return get_OrganCMNIST(get_handler(name), args_task, args)
	elif name == 'OrganSMNIST':
		return get_OrganSMNIST(get_handler(name), args_task, args)
	elif name == 'BloodMNIST':
		return get_BloodMNIST(get_handler(name), args_task, args)
	else:
		raise NotImplementedError

# net
def get_net_mae(name, args_task, device, args):
	if name == 'DermaMNIST':
		return Net_MAE(Adapter, args_task, device, args)
	elif name == 'BreastMNIST':
		return Net_MAE(Adapter, args_task, device, args)
	elif name == 'PathMNIST':
		return Net_MAE(Adapter, args_task, device, args)
	elif name == 'BloodMNIST':
		return Net_MAE(Adapter, args_task, device, args)
	elif name == 'ChestMNIST':
		return Net_MAE(Adapter, args_task, device, args)
	elif name == 'OCTMNIST':
		return Net_MAE(Adapter, args_task, device, args)
	elif name == 'PneumoniaMNIST':
		return Net_MAE(Adapter, args_task, device, args)
	elif name == 'TissueMNIST':
		return Net_MAE(Adapter, args_task, device, args)
	elif name == 'OrganAMNIST':
		return Net_MAE(Adapter, args_task, device, args)
	elif name == 'OrganCMNIST':
		return Net_MAE(Adapter, args_task, device, args)
	elif name == 'OrganSMNIST':
		return Net_MAE(Adapter, args_task, device, args)
	elif name == 'CIFAR10':
		return Net_MAE(Adapter, args_task, device, args)
	else:
		raise NotImplementedError

def get_net(name, args_task, device, args):
	if name == 'CIFAR10':
		return Net(CIFAR10_Net, args_task, device, args)
	elif 'PneumoniaMNIST' in name:
		return Net(MedMNIST_Net, args_task, device, args)
	elif 'DermaMNIST' in name:
		return Net(MedMNIST_Net, args_task, device, args)
	elif 'BreastMNIST' in name:
		return Net(MedMNIST_Net, args_task, device, args)
	elif 'BloodMNIST' in name:
		return Net(MedMNIST_Net, args_task, device, args)
	elif 'OrganCMNIST' in name:
		return Net(MedMNIST_Net, args_task, device, args)
	elif 'OrganSMNIST' in name:
		return Net(MedMNIST_Net, args_task, device, args)
	elif 'OrganAMNIST' in name:
		return Net(MedMNIST_Net, args_task, device, args)
	elif 'ChestMNIST' in name:
		return Net(MedMNIST_Net, args_task, device, args)
	else:
		raise NotImplementedError

def get_params(name):
	return args_pool[name]

#strategy

def get_strategy(STRATEGY_NAME, dataset, net, args_input, args_task):
	if STRATEGY_NAME == 'RandomSampling':
		return RandomSampling(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'LeastConfidence':
		return LeastConfidence(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'MarginSampling':
		return MarginSampling(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'EntropySampling':
		return EntropySampling(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'LeastConfidenceDropout':
		return LeastConfidenceDropout(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'MarginSamplingDropout':
		return MarginSamplingDropout(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'EntropySamplingDropout':
		return EntropySamplingDropout(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'KCenterGreedy':
		return KCenterGreedy(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'KCenterGreedyPCA':
		return KCenterGreedyPCA(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'BALDDropout':
		return BALDDropout(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'MeanSTD':
		return MeanSTD(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'BadgeSampling':
		return BadgeSampling(dataset, net, args_input, args_task)
	else:
		raise NotImplementedError



"""
Utils for Results 
"""
class Logger(object):
	def __init__(self, filename="Default.log"):
		self.terminal = sys.stdout
		self.log = open(filename, "a")

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)

	def flush(self):
		pass

def get_mean_stddev(datax):
	return round(np.mean(datax),4),round(np.std(datax),4)

def get_aubc(quota, bsize, resseq):
	# it is equal to use np.trapz for calculation
	ressum = 0.0
	if quota % bsize == 0:
		for i in range(len(resseq)-1):
			ressum = ressum + (resseq[i+1] + resseq[i]) * bsize / 2

	else:
		for i in range(len(resseq)-2):
			ressum = ressum + (resseq[i+1] + resseq[i]) * bsize / 2
		k = quota % bsize
		ressum = ressum + ((resseq[-1] + resseq[-2]) * k / 2)
	ressum = round(ressum / quota,3)
	
	return ressum



def eval_skl_model(pre, pre_proba,labels):
    acc_te = accuracy_score(pre,labels)
    num_class = pre_proba.shape[-1]
    if labels.max() > 2:
        auc_te = roc_auc_score(np.eye(num_class)[labels],pre_proba,multi_class='ovr',average='micro')
    else:
        auc_te = roc_auc_score(labels,pre_proba[:,1])

    return acc_te,auc_te

def eval_round(strategy,dataset):
	preds_tr_lab = strategy.predict(dataset.get_labeled_data()[-1])
	preds_tr_unl = strategy.predict(dataset.get_unlabeled_data()[-1])
	preds_te = strategy.predict(dataset.get_test_data())

	preds_proba_tr_lab = strategy.predict_prob(dataset.get_labeled_data()[-1])
	preds_proba_tr_unl = strategy.predict_prob(dataset.get_unlabeled_data()[-1])
	preds_proba_te = strategy.predict_prob(dataset.get_test_data())

	y_te = dataset.Y_test.numpy()
	y_tr_lab = dataset.Y_train[dataset.get_labeled_data()[0]].numpy()
	y_tr_unl = dataset.Y_train[dataset.get_unlabeled_data()[0]].numpy()
	
	acc_tr_lab,auc_tr_lab = eval_skl_model(preds_tr_lab, preds_proba_tr_lab, y_tr_lab)
	acc_tr_unl,auc_tr_unl = eval_skl_model(preds_tr_unl, preds_proba_tr_unl, y_tr_unl)
	acc_te,auc_te = eval_skl_model(preds_te, preds_proba_te, y_te)
	return acc_tr_lab,auc_tr_lab,acc_tr_unl,auc_tr_unl,acc_te,auc_te 


def render_res_file(args_input):
	dir1 = 'Data_'+str(args_input.dataset_name)+'_Strategy_'+str(args_input.ALstrategy)+'_Model_'+str(args_input.model)
	dir2 = 'Seed_'+str(args_input.seed)+'_Query_'+str(args_input.batch)+'_Round_'+str(args_input.round)+'_Init_'+str(args_input.initseed)
	file_path = os.path.join(args_input.out_path,dir1,dir2)
	if os.path.exists(file_path) == False:
		os.makedirs(file_path)
	return file_path


def render_res_file_mae(args_input):
	dir_mae = "MAE_"+str(args_input.model_mae) + '_' + str(args_input.pretrained_data) + '_' + str(args_input.adapter)
	dir_mae_2 = "Tau_"+str(args_input.tau)
	dir1 = 'Data_'+str(args_input.dataset_name)+'_Strategy_'+str(args_input.ALstrategy)
	dir2 = 'Seed_'+str(args_input.seed)+'_Query_'+str(args_input.batch)+'_Round_'+str(args_input.round)+'_Init_'+str(args_input.initseed)
	if args_input.tau != 0:
		semi_dir = '_num_' + str(args_input.num_unl) + '_decay_' + str(args_input.semi_decay) + '_' + str(args_input.semi_loss) + '_' + str(args_input.unl_sample) 
		dir_mae_2 = dir_mae_2 + semi_dir
	
	if args_input.aug_strategy != 'No':
		dir_mae_2 = dir_mae_2 + '_' + str(args_input.aug_strategy)
		if args_input.aug_strategy == 'Self':
			dir_mae_3 = 'Local_' + args_input.local_strategy + '_' + str(args_input.local_ratio)
			if args_input.label_smooth_local:
				dir_mae_3 = dir_mae_3 + '_LM'
		elif args_input.aug_strategy == 'SamplePatchAug':
			dir_mae_3 = 'Local_' + args_input.local_strategy + '_' + str(args_input.local_ratio) + '_topk_'+  str(args_input.sample_top_k) +'_trade_' + str(args_input.raw_rep_ratio)
			if args_input.label_smooth_local:
				dir_mae_3 = dir_mae_3 + '_LM'
    
	if args_input.aug_strategy == 'Self':
		file_path = os.path.join(args_input.out_path,dir_mae,dir_mae_2,dir_mae_3,dir1,dir2)
	elif args_input.aug_strategy == 'SamplePatchAug':
		file_path = os.path.join(args_input.out_path,dir_mae,dir_mae_2,dir_mae_3,dir1,dir2)
	else:
		file_path = os.path.join(args_input.out_path,dir_mae,dir_mae_2,dir1,dir2)
		

	if os.path.exists(file_path) == False:
		os.makedirs(file_path)
	
	return file_path


"""
For Distance Matrix 
"""
def render_res_file_mae_dis_matrix(args_input,create=True):
	dir_mae = "MAE_"+str(args_input.model_mae) + '_' + str(args_input.pretrained_data) + '_' + 'Data_'+str(args_input.dataset_name)
	if args_input.use_global_avg is True:
		dir_mae += '_global_avg'
	file_path = os.path.join(args_input.out_path,'dis_matrix',dir_mae)
    
	if create:
		if os.path.exists(file_path) == False:
			os.makedirs(file_path)
	return file_path

def get_dis_matrix(args_input):
	dis_mat = torch.load(os.path.join(args_input.dis_matrix_path,'dis_mat.pt'),map_location=torch.device('cpu'))
	dis_indices = torch.load(os.path.join(args_input.dis_matrix_path,'dis_indices.pt'),map_location=torch.device('cpu'))	
	return dis_mat,dis_indices
