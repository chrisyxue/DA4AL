import numpy as np
import torch
from utils import get_dataset, get_net, get_strategy, eval_round, render_res_file
from pprint import pprint
torch.set_printoptions(profile='full')

import sys
import os
import re
import random
import math
import datetime

import arguments
from parameters import *
from utils import *
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd


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
args_input = arguments.get_args()
NUM_QUERY = args_input.batch
NUM_INIT_LB = args_input.initseed
NUM_ROUND = args_input.round
DATA_NAME = args_input.dataset_name
STRATEGY_NAME = args_input.ALstrategy
args_input.quota = NUM_ROUND * NUM_QUERY
SEED = args_input.seed

# os.environ['TORCH_HOME']='./basicmodel'
os.environ['TORCH_HOME'] = args_input.pretrained_model_path
os.environ["CUDA_VISIBLE_DEVICES"] = str(args_input.gpu)

# fix random seed
fix_seed(SEED)

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# recording
# sys.stdout = Logger(os.path.abspath('') + '/logfile/' + DATA_NAME+ '_'  + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(NUM_INIT_LB) +  '_' + str(args_input.quota) + '_normal_log.txt')
# warnings.filterwarnings('ignore')

# start experiment
iteration = args_input.iteration
all_acc_te = []
all_acc_tr_lab = []
all_acc_tr_unl = []
all_auc_te = []
all_auc_tr_lab = []
all_auc_tr_unl = []

acq_time = []

# repeate # iteration trials
while (iteration > 0): 
	fix_seed(SEED+iteration)
	iteration = iteration - 1
	
	# data, network, strategy
	args_task = args_pool[DATA_NAME]
	dataset = get_dataset(args_input.dataset_name, args_task, args_input)				# load dataset
	
	net = get_net(args_input.dataset_name, args_task, device, args_input)			# load network
	strategy = get_strategy(args_input.ALstrategy, dataset, net, args_input, args_task)  # load strategy

	start = datetime.datetime.now()
	# generate initial labeled pool
	dataset.initialize_labels(args_input.initseed)

	#record acc performance
	acc_tr_lab_lst = np.zeros(NUM_ROUND + 1)
	acc_tr_unl_lst = np.zeros(NUM_ROUND + 1)
	acc_te_lst = np.zeros(NUM_ROUND + 1)
	auc_tr_lab_lst = np.zeros(NUM_ROUND + 1)
	auc_tr_unl_lst = np.zeros(NUM_ROUND + 1)
	auc_te_lst = np.zeros(NUM_ROUND + 1)

	# only for special cases that need additional data
	new_X = torch.empty(0)
	new_Y = torch.empty(0)
		
	# print info
	print(DATA_NAME)
	print('RANDOM SEED {}'.format(SEED))
	print(type(strategy).__name__)
	
	# round 0 accuracy
	if NUM_INIT_LB == 0:
		print('No Initial Samples')
		acc_tr_lab_lst[0] = 0
		acc_tr_unl_lst[0] = 0
		acc_te_lst[0] = 0
		auc_tr_lab_lst[0] = 0
		auc_tr_unl_lst[0] = 0
		auc_te_lst[0] = 0
	else:
		strategy.train()
		preds = strategy.predict(dataset.get_test_data())
		acc_tr_lab,auc_tr_lab,acc_tr_unl,auc_tr_unl,acc_te,auc_te = eval_round(strategy,dataset)
		acc_tr_lab_lst[0] = acc_tr_lab
		acc_tr_unl_lst[0] = acc_tr_unl
		acc_te_lst[0] = acc_te
		auc_tr_lab_lst[0] = auc_tr_lab
		auc_tr_unl_lst[0] = auc_tr_unl
		auc_te_lst[0] = auc_te
		print('Round 0\ntesting accuracy {}'.format(acc_te_lst[0]))
		print('\n')
	
	# round 1 to rd
	for rd in range(1, NUM_ROUND+1):
		print('Round {}'.format(rd))
		high_confident_idx = []
		high_confident_pseudo_label = []
		# query
		q_idxs = strategy.query(NUM_QUERY)
	
		# update
		strategy.update(q_idxs)

		# train
		strategy.train()
        
        # round train/test acc and auc
		acc_tr_lab,auc_tr_lab,acc_tr_unl,auc_tr_unl,acc_te,auc_te = eval_round(strategy,dataset)
		acc_tr_lab_lst[rd] = acc_tr_lab
		acc_tr_unl_lst[rd] = acc_tr_unl
		acc_te_lst[rd] = acc_te
		auc_tr_lab_lst[rd] = auc_tr_lab
		auc_tr_unl_lst[rd] = auc_tr_unl
		auc_te_lst[rd] = auc_te
        
		print('testing accuracy {}'.format(acc_te_lst[rd]))
		print('testing auc {}'.format(auc_te_lst[rd]))
		print('\n')

		#torch.cuda.empty_cache()
	
	# print results
	print('SEED {}'.format(SEED))
	print(type(strategy).__name__)

	all_acc_te.append(acc_te_lst)
	all_acc_tr_lab.append(acc_tr_lab_lst)
	all_acc_tr_unl.append(acc_tr_unl_lst)

	all_auc_te.append(auc_te_lst)
	all_auc_tr_lab.append(auc_tr_lab_lst)
	all_auc_tr_unl.append(auc_tr_unl_lst)

	#save model
	end = datetime.datetime.now()
	acq_time.append(round(float((end-start).seconds),3))
	
# cal mean & standard deviation
mean_accm_te, stddev_accm_te = mean_var_aubc(all_acc_te)
mean_accm_tr_lab, stddev_accm_tr_lab = mean_var_aubc(all_acc_tr_lab)
mean_accm_tr_unl, stddev_accm_tr_unl = mean_var_aubc(all_acc_tr_unl)

mean_aucm_te, stddev_aucm_te = mean_var_aubc(all_auc_te)
mean_aucm_tr_lab, stddev_aucm_tr_lab = mean_var_aubc(all_auc_tr_lab)
mean_aucm_tr_unl, stddev_aucm_tr_unl = mean_var_aubc(all_auc_tr_unl)


mean_acc_te, stddev_acc_te = mean_var(all_acc_te)
mean_acc_tr_lab, stddev_acc_tr_lab = mean_var(all_acc_tr_lab)
mean_acc_tr_unl, stddev_acc_tr_unl = mean_var(all_acc_tr_unl)

mean_auc_te, stddev_auc_te = mean_var(all_auc_te)
mean_auc_tr_lab, stddev_auc_tr_lab = mean_var(all_auc_tr_lab)
mean_auc_tr_unl, stddev_auc_tr_unl = mean_var(all_auc_tr_unl)

mean_time, stddev_time = get_mean_stddev(acq_time)

# store data for accm/aucm
data_res_mean = pd.DataFrame([[mean_accm_te,mean_accm_tr_lab,mean_accm_tr_unl,mean_aucm_te,mean_aucm_tr_lab,mean_aucm_tr_unl]],columns=['accm_te','accm_tr_lab','accm_tr_unl','aucm_te','aucm_tr_lab','aucm_tr_unl'])
data_res_std = pd.DataFrame([[stddev_accm_te,stddev_accm_tr_lab,stddev_accm_tr_unl,stddev_aucm_te,stddev_aucm_tr_lab,stddev_aucm_tr_unl]],columns=['accm_te','accm_tr_lab','accm_tr_unl','aucm_te','aucm_tr_lab','aucm_tr_unl'])

# store data for acc/auc
res_curve_mean = np.array([mean_acc_te.tolist(),mean_acc_tr_lab.tolist(),mean_acc_tr_unl.tolist(),mean_auc_te.tolist(),mean_auc_tr_lab.tolist(),mean_auc_tr_unl.tolist()]).T
res_curve_mean = pd.DataFrame(res_curve_mean,columns=['acc_te','acc_tr_lab','acc_tr_unl','auc_te','auc_tr_lab','auc_tr_unl'])
res_curve_std = np.array([stddev_acc_te.tolist(),stddev_acc_tr_lab.tolist(),stddev_acc_tr_unl.tolist(),stddev_auc_te.tolist(),stddev_auc_tr_lab.tolist(),stddev_auc_tr_unl.tolist()]).T
res_curve_std = pd.DataFrame(res_curve_std,columns=['acc_te','acc_tr_lab','acc_tr_unl','auc_te','auc_tr_lab','auc_tr_unl'])

# save Results
file_path = render_res_file(args_input)
res_curve_mean.to_csv(os.path.join(file_path,'res_curve_mean.csv'))
res_curve_std.to_csv(os.path.join(file_path,'res_curve_std.csv'))
data_res_mean.to_csv(os.path.join(file_path,'res_mean.csv'))
data_res_std.to_csv(os.path.join(file_path,'res_std.csv'))
res_curve_std.to_csv(os.path.join(file_path,'time.csv'))

