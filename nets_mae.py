import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.autograd import Variable
from copy import deepcopy
from tqdm import tqdm
import torch.nn.init as init
import pdb
import mae.models_mae as models_mae
import mae.misc as misc
from torch.utils.data.dataset import random_split
from mae.semi_loss import EntropyLoss, L2RelationLoss, L2RelationNormLoss
from mae.data_aug import get_augmentor
from mae.data_aug import get_tensor_augdata
import pdb
from arguments import WEIGHT_PATH_BASE
# from utils import get_dis_matrix


TRANSFORM_AUG = ['Normal','Auto','Rand']
TENSOR_AUG = ['Self','MixUp','CutMix','Sample','SamplePatchAug','ColorPatchAug','FlipPatchAug','MaskPatchAug','AutoPatchAug','RandPatchAug','GaussianPatchAug']

def random_sample(dataset, sample_size, batch_size=1, shuffle=False):

  # Split the dataset into a number of chunks
  chunks = random_split(dataset, [sample_size, len(dataset) - sample_size])
  # Get the first chunk, which is the sample
  data_sample = chunks[0]
  data_loader = DataLoader(data_sample, batch_size=batch_size, shuffle=shuffle)


  return torch.cat([batch[0] for batch in data_loader]),torch.cat([batch[1] for batch in data_loader])



class Net_MAE:
	def __init__(self, adapter, params, device, args):
		# self.net = net
		self.adapter = adapter
		self.params = params
		self.device = device
		self.args = args
		self.load_mae_model()


	def load_mae_model(self):
		model = models_mae.__dict__[self.args.model_mae](norm_pix_loss=self.args.norm_pix_loss,img_size=self.params['img_size'])
		if self.args.pretrained_data == 'Source':
			self.args.resume = self.args.checkpoint_source
			checkpoint_type='MAE'
		elif self.args.pretrained_data == 'Source_FT':
			self.args.resume = self.args.checkpoint_sourceft
			checkpoint_type='MAE'
		elif self.args.pretrained_data == 'General':
			self.args.resume = 'https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth'
			checkpoint_type='MAE'
		elif self.args.pretrained_data == 'MoCo':
			self.args.resume = WEIGHT_PATH_BASE[self.args.pretrained_data]
			checkpoint_type = 'MoCo'
			assert(self.args.aug_strategy != 'Self')
		elif self.args.pretrained_data == 'CXR':
			self.args.resume = WEIGHT_PATH_BASE[self.args.pretrained_data]
			checkpoint_type = 'CXR'
			assert(self.args.aug_strategy != 'Self')
		elif self.args.pretrained_data == 'DINO':
			self.args.resume = WEIGHT_PATH_BASE[self.args.pretrained_data]
			checkpoint_type = 'DINO'
			assert(self.args.aug_strategy != 'Self')
			# self.args.resume = '/home/zhiyu/models/vit/mae_pretrain_vit_base.pth'
		else:
			raise ValueError('No pretrained data like '+self.args.pretrained_data)
		
		misc.load_pretrained_model(self.args, model, checkpoint_type=checkpoint_type)
		model.eval()
		self.model_mae = model
		self.mae_params = {}
		self.mae_params['patch_size'] = self.model_mae.patch_size
		self.mae_params['embed_dim'] = self.model_mae.embed_dim 
		self.mae_params['img_size'] = self.model_mae.img_size
		self.mae_params['num_patches'] = self.model_mae.patch_embed.num_patches
		self.model_mae.to(self.device)
	
	def mae_encode(self,x):
		with torch.no_grad():
			feats,_,_ = self.model_mae.forward_encoder(x, mask_ratio=0)
			if self.args.use_global_avg:
				feats = feats[:, 1:, :].mean(dim=1)
			else:
				feats = feats[:, 0]	
		return feats
	
	def get_aug_data_from_loaders(self,sample_aug_data_lst):
		aug_data_lst = []
		for loader in sample_aug_data_lst:
			aug_data = []
			for batch_idx, (x, y, idxs) in enumerate(loader):
				aug_data.append(x)
			aug_data = torch.concat(aug_data)
			aug_data_lst.append(aug_data)
		return aug_data_lst
		
	def train_sup(self, data, unl_data, sample_aug_data_lst=None):
		if sample_aug_data_lst is not None:
			assert(len(sample_aug_data_lst) == len(data))
			sample_aug_data_lst = [DataLoader(data_aug, shuffle=True, batch_size=len(data_aug)) for data_aug in sample_aug_data_lst]
			sample_aug_data_lst = self.get_aug_data_from_loaders(sample_aug_data_lst)

		n_epoch = self.params['n_epoch']
		# init adapter
		self.clf = self.adapter(embed_dim=self.mae_params['embed_dim'],num_classes=self.params['num_class'],adapter_type=self.args.adapter).to(self.device)
		self.clf.train()
		if self.params['optimizer'] == 'Adam':
			optimizer = optim.Adam(self.clf.parameters(), **self.params['optimizer_args'])
		elif self.params['optimizer'] == 'SGD':
			optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])
		else:
			raise NotImplementedError

		# data augmentation on transform (e.g. AutoAug, RandAug)
		if self.args.aug_strategy in TRANSFORM_AUG:
			trans_aug = get_augmentor(args_input=self.args,args_task=self.params)			
			data.get_aug_trans(trans_aug)

		loader = DataLoader(data, shuffle=True, **self.params['loader_tr_args'])
		for epoch in range(1, n_epoch+1):
			for batch_idx, (x, y, idxs) in enumerate(loader):
				"""
				TODO:
				Add DA Here
				"""
				# print(len(data))
				if self.args.aug_strategy in TRANSFORM_AUG:
					# Lab Data
					x_ori,x_aug = x[0],x[1]
					x_ori,x_aug = x_ori.to(self.device),x_aug.to(self.device)
					y = y.to(self.device)
					y = torch.concat([y,y])
					# For BN layer:
					if x_ori.shape[0] == 1:
						continue
					feats_ori = self.mae_encode(x_ori)
					feats_aug = self.mae_encode(x_aug)
					feats = torch.concat([feats_ori,feats_aug],dim=0)
				elif self.args.aug_strategy in TENSOR_AUG:
					x_ori = x.to(self.device)
					y_ori = y.to(self.device)
					# y_ori = F.one_hot(y_ori,num_classes=self.params['num_class'])
					# For BN layer:
					if x_ori.shape[0] == 1:
						continue
					if 'Sample' in self.args.aug_strategy and sample_aug_data_lst is not None:
						sample_aug_data_batch = [sample_aug_data_lst[i].unsqueeze(0).to(self.device) for i in idxs.tolist()]
						assert(len(sample_aug_data_batch) == idxs.shape[0])
					else:
						sample_aug_data_batch = None
					x_aug, y_aug = get_tensor_augdata(args_input=self.args,args_task=self.params,
				       				images=x_ori,labels=y_ori,model_mae=self.model_mae,
									adapter=self.clf,sample_aug_data_batch=sample_aug_data_batch)	
					
					feats_ori = self.mae_encode(x_ori)
					feats_aug = self.mae_encode(x_aug)
					feats = torch.concat([feats_ori,feats_aug],dim=0)
					y_ori = F.one_hot(y_ori,num_classes=self.params['num_class'])
					feats = torch.concat([feats_ori,feats_aug],dim=0)
					y = torch.concat([y_ori,y_aug],dim=0)
					y = y.type(torch.float32)
				else:
					x, y = x.to(self.device), y.to(self.device)
					# For BN layer:
					if x.shape[0] == 1:
						continue
					feats = self.mae_encode(x)

				optimizer.zero_grad()
				out, e1 = self.clf(feats)
				# pdb.set_trace()
				loss = F.cross_entropy(out, y)
				loss.backward()
				optimizer.step()
	
	def sample_unl_data(self,unl_data):
		# Sample Unl Data
		if self.args.unl_sample == 'Random':
			x_unl,y_unl = random_sample(unl_data,self.args.num_unl)
		else:
			raise ValueError('No Unlabelled Sample Method named '+self.args.unl_sample)
		return x_unl,y_unl

	def train_semi(self, data, unl_data):
		n_epoch = self.params['n_epoch']
		self.clf = self.adapter(embed_dim=self.mae_params['embed_dim'],num_classes=self.params['num_class'],adapter_type=self.args.adapter).to(self.device)
		self.clf.train()
		if self.params['optimizer'] == 'Adam':
			optimizer = optim.Adam(self.clf.parameters(), **self.params['optimizer_args'])
		elif self.params['optimizer'] == 'SGD':
			optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])
		else:
			raise NotImplementedError
		# data augmentation on transform
		if self.args.aug_strategy in TRANSFORM_AUG:
			trans_aug = get_augmentor(args_input=self.args,args_task=self.params)			
			data.get_aug_trans(trans_aug)

		loader = DataLoader(data, shuffle=True, **self.params['loader_tr_args'])
		# unl_dataset = DataLoader(unl_data, shuffle=True, batch_size=self.args.num_unl).datset
        
		for epoch in range(1, n_epoch+1):
			"""
				TODO:
				Add DA Here
			"""
			for batch_idx, (x, y, idxs) in enumerate(loader):
				if self.args.aug_strategy in TRANSFORM_AUG:
					# Lab Data
					x_ori,x_aug = x[0],x[1]
					x_ori,x_aug = x_ori.to(self.device),x_aug.to(self.device)
					y = y.to(self.device)
					y = torch.concat([y,y])

					# For BN layer:
					if x_ori.shape[0] == 1:
						continue
					feats_ori = self.mae_encode(x_ori)
					feats_aug = self.mae_encode(x_aug)
					feats = torch.concat([feats_ori,feats_aug],dim=0)
					
					# Unl Data
					x_unl, y_unl = self.sample_unl_data(unl_data)
					x_unl, y_unl = x_unl.to(self.device), y_unl.to(self.device)
					feats_unl = self.mae_encode(x_unl)
					out_unl, e1_unl = self.clf(feats_unl)
				elif self.args.aug_strategy in TENSOR_AUG:

					# Lab Data
					x_ori = x.to(self.device)
					y_ori = y.to(self.device)
					# For BN layer:
					if x_ori.shape[0] == 1:
						continue
					x_aug, y_aug = get_tensor_augdata(args_input=self.args,args_task=self.params,images=x_ori,labels=y_ori,model_mae=self.model_mae,adapter=self.clf)	
					
					feats_ori = self.mae_encode(x_ori)
					feats_aug = self.mae_encode(x_aug)
					feats = torch.concat([feats_ori,feats_aug],dim=0)
					y_ori = F.one_hot(y_ori,num_classes=self.params['num_class'])
					feats = torch.concat([feats_ori,feats_aug],dim=0)
					y = torch.concat([y_ori,y_aug],dim=0)

					# Unl Data
					x_unl, y_unl = self.sample_unl_data(unl_data)
					x_unl, y_unl = x_unl.to(self.device), y_unl.to(self.device)
					feats_unl = self.mae_encode(x_unl)
					out_unl, e1_unl = self.clf(feats_unl)
				else:
					# Lab Data
					x, y = x.to(self.device), y.to(self.device)
					feats = self.mae_encode(x)
					# For BN layer:
					if x.shape[0] == 1:
						continue

					# Unl Data
					x_unl, y_unl = self.sample_unl_data(unl_data)
					x_unl, y_unl = x_unl.to(self.device), y_unl.to(self.device)
					feats_unl = self.mae_encode(x_unl)
					out_unl, e1_unl = self.clf(feats_unl)
	

				optimizer.zero_grad()
				out, e1 = self.clf(feats)
				loss = F.cross_entropy(out, y)

				if self.args.semi_loss == 'L2Cons':
					semi_loss = F.mse_loss(e1_unl,feats_unl)
				elif self.args.semi_loss == 'EntMin':
					semi_loss = EntropyLoss(out_unl)
				elif self.args.semi_loss == 'L2Relation':
					semi_loss = L2RelationLoss(e1,feats,e1_unl,feats_unl)
				elif self.args.semi_loss == 'L2RelationNorm':
					semi_loss = L2RelationNormLoss(e1,feats,e1_unl,feats_unl)
				else:
					raise ValueError('No Unl Loss named '+self.args.semi_loss)
                
				tau = self.args.tau*(self.args.semi_decay**epoch)
				loss = loss + tau*semi_loss
				loss.backward()
				optimizer.step()




	def train(self, data, unl_data,sample_aug_data_lst=None):
		if self.args.tau != 0:
			self.train_semi(data, unl_data,sample_aug_data_lst=sample_aug_data_lst)
		else:
			self.train_sup(data, unl_data,sample_aug_data_lst=sample_aug_data_lst)

	def predict(self, data):
		self.clf.eval()
		preds = torch.zeros(len(data), dtype=data.Y.dtype)
		loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])

		with torch.no_grad():
			for x, y, idxs in loader:
				x, y = x.to(self.device), y.to(self.device)
				x = self.mae_encode(x)
				out, e1 = self.clf(x)
				pred = out.max(1)[1]
				preds[idxs] = pred.cpu()
		return preds
	
	def predict_prob(self, data, num_class=None):
		self.clf.eval()
		loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
		if num_class == None:
			probs = torch.zeros([len(data), len(np.unique(data.Y))])
		else:
			probs = torch.zeros([len(data), num_class])
		with torch.no_grad():
			for x, y, idxs in loader:
				x, y = x.to(self.device), y.to(self.device)
				x = self.mae_encode(x)
				out, e1 = self.clf(x)
				prob = F.softmax(out, dim=1)
				probs[idxs] = prob.cpu()
		return probs
	
	def predict_prob_dropout(self, data, n_drop=10, num_class=None):
		self.clf.train()
		loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
		if num_class == None:
			probs = torch.zeros([len(data), len(np.unique(data.Y))])
		else:
			probs = torch.zeros([len(data), num_class])
		for i in range(n_drop):
			with torch.no_grad():
				for x, y, idxs in loader:
					x, y = x.to(self.device), y.to(self.device)
					# For BN layer:
					if x.shape[0] == 1:
						continue
					x = self.mae_encode(x)
					out, e1 = self.clf(x)
					prob = F.softmax(out, dim=1)
					probs[idxs] += prob.cpu()
		probs /= n_drop
		return probs
	
	def predict_prob_dropout_split(self, data, n_drop=10, num_class=None):
		self.clf.train()
		loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
		if num_class == None:
			probs = torch.zeros([n_drop, len(data), len(np.unique(data.Y))])
		else:
			probs = torch.zeros([n_drop, len(data), num_class])
		for i in range(n_drop):
			with torch.no_grad():
				for x, y, idxs in loader:
					x, y = x.to(self.device), y.to(self.device)
					# For BN layer:
					if x.shape[0] == 1:
						continue
					x = self.mae_encode(x)
					out, e1 = self.clf(x)
					prob = F.softmax(out, dim=1)
					probs[i][idxs] += F.softmax(out, dim=1).cpu()
		return probs
	
	def get_model(self):
		return self.clf

	def get_embeddings(self, data):
		self.clf.eval()
		embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
		loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
		with torch.no_grad():
			for x, y, idxs in loader:
				x, y = x.to(self.device), y.to(self.device)
				x = self.mae_encode(x)
				out, e1 = self.clf(x)
				embeddings[idxs] = e1.cpu()
		return embeddings
	
	def get_grad_embeddings(self, data):
		self.clf.eval()
		embDim = self.clf.get_embedding_dim()
		nLab = self.params['num_class']
		embeddings = np.zeros([len(data), embDim * nLab])

		loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
		with torch.no_grad():
			for x, y, idxs in loader:
				x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
				x = self.mae_encode(x)
				cout, out = self.clf(x)
				out = out.data.cpu().numpy()
				batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
				maxInds = np.argmax(batchProbs,1)
				for j in range(len(y)):
					for c in range(nLab):
						if c == maxInds[j]:
							embeddings[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c]) * -1.0
						else:
							embeddings[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c]) * -1.0

		return embeddings

def kaiming_init(m):
		if isinstance(m, (nn.Linear, nn.Conv2d)):
				init.kaiming_normal(m.weight)
				if m.bias is not None:
						m.bias.data.fill_(0)
		elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
				m.weight.data.fill_(1)
				if m.bias is not None:
						m.bias.data.fill_(0)

def normal_init(m, mean, std):
	if isinstance(m, (nn.Linear, nn.Conv2d)):
		m.weight.data.normal_(mean, std)
		if m.bias.data is not None:
				m.bias.data.zero_()
	elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
		m.weight.data.fill_(1)
		if m.bias.data is not None:
				m.bias.data.zero_()


class PrintLayer(nn.Module):
	def __init__(self):
		super(PrintLayer, self).__init__()
	
	def forward(self, x):
		# Do your print / debug stuff here
		print('p',x.shape)      #print(x.shape)
		return x
