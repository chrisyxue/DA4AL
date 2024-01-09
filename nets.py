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
import analysis
from analysis.ssl_models.vit import vit_base_load_weights
from mae.models_adapter import Adapter

class Net:
    def __init__(self, net, params, device, args):
        self.net = net
        self.params = params
        self.device = device
        self.args = args
		
    def train(self, data):
		
        n_epoch = self.params['n_epoch']

        dim = data.X.shape[1:]
        self.clf = self.net(dim = dim, pretrained = self.params['pretrained'], num_classes = self.params['num_class'], model=self.args.model).to(self.device)
        self.clf.train()
		
        if '_ft' in self.args.model:
          self.clf.classifier = Adapter(embed_dim=self.clf.dim,num_classes=self.clf.num_classes).to(self.device)
          self.clf.use_adapter = True
        if self.params['optimizer'] == 'Adam':
            optimizer = optim.Adam(self.clf.parameters(), **self.params['optimizer_args'])
            if '_ft' in self.args.model:
              optimizer = optim.Adam(self.clf.classifier.parameters(), **self.params['optimizer_args'])
        elif self.params['optimizer'] == 'SGD':
            optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])
            if '_ft' in self.args.model:
              optimizer = optim.SGD(self.clf.classifier.parameters(), **self.params['optimizer_args'])
        else:
            raise NotImplementedError

        loader = DataLoader(data, shuffle=True, **self.params['loader_tr_args'])
        for epoch in range(1, n_epoch+1):
            for batch_idx, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                if x.shape[0] == 1 and  '_ft' in self.args.model:
                  continue
                optimizer.zero_grad()
                out, e1 = self.clf(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()

    def predict(self, data):
        self.clf.eval()
        preds = torch.zeros(len(data), dtype=data.Y.dtype)
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])

        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
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

class MedMNIST_Net(nn.Module):
	def __init__(self, dim = 224 * 224, pretrained=False, num_classes = 10, model='resnet18'):
		super().__init__()
		if model == 'resnet18':
			self.base_model = models.resnet18(pretrained=pretrained)
		elif model == 'resnet34':
			self.base_model = models.resnet34(pretrained=pretrained)
		elif model == 'resnet50':
			self.base_model = models.resnet50(pretrained=pretrained)
		elif model == 'resnet18_pretrain' or model == 'resnet18_pretrain_ft':
			self.base_model = models.resnet18(pretrained=True)
		elif model == 'resnet34_pretrain' or model == 'resnet34_pretrain_ft':
			self.base_model = models.resnet34(pretrained=True)
		elif model == 'resnet50_pretrain' or model == 'resnet50_pretrain_ft':
			self.base_model = models.resnet50(pretrained=True)
		elif model == 'vgg16':
			self.base_model = models.vgg16(pretrained=pretrained)
		elif model == 'alexnet':
			self.base_model = models.alexnet(pretrained=pretrained)
		elif model == 'mobilenet':
			self.base_model = models.mobilenet_v2(pretrained=pretrained)
		else:
			raise ValueError('No Model Named '+str(model))
	
		if 'resnet' in model:
			features_tmp = nn.Sequential(*list(self.base_model.children())[:-1])
			features_tmp[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
			self.features = nn.Sequential(*list(features_tmp))
			self.classifier = nn.Linear(self.base_model.fc.in_features,num_classes)
			self.dim = self.base_model.fc.in_features
		elif 'vgg' in model:
			features_tmp = nn.Sequential(*list(self.base_model.children())[:-1])
			# features_tmp[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
			self.features = nn.Sequential(*list(features_tmp))
			self.classifier = nn.Linear(self.base_model.classifier[0].in_features,num_classes)
			self.dim = self.base_model.classifier[0].in_features
		elif 'alexnet' in model or 'mobilenet' in model:
			features_tmp = nn.Sequential(*list(self.base_model.children())[:-1])
			self.features = nn.Sequential(*list(features_tmp))
			self.classifier = nn.Linear(self.base_model.classifier[1].in_features,num_classes)
			self.dim = self.base_model.classifier[1].in_features
		else:
			raise ValueError('No such model named '+ model)
		self.num_classes = num_classes
		self.use_adapter = False
    
	def forward(self, x):
		feature  = self.features(x)
		x = feature.view(feature.size(0), -1)
		if self.use_adapter:
			output = self.classifier.forward_out(x)
		else:
			output = self.classifier(x)
		return output, x
	
	def get_embedding_dim(self):
		return self.dim
	
	# def init_adapter(self):
	# 	self.classifier = Adapter(embed_dim=self.dim,num_classes=self.num_classes)

class CIFAR10_Net(nn.Module):
	def __init__(self, dim = 224 * 224, pretrained=False, num_classes = 10,  model='resnet18'):
		super().__init__()
		if model == 'resnet18':
			self.base_model = models.resnet18(pretrained=pretrained)
		elif model == 'resnet34':
			self.base_model = models.resnet34(pretrained=pretrained)
		elif model == 'resnet50':
			self.base_model = models.resnet50(pretrained=pretrained)
		else:
			raise ValueError('No Model Named '+str(model))
		features_tmp = nn.Sequential(*list(self.base_model.children())[:-1])
		features_tmp[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.features = nn.Sequential(*list(features_tmp))
		self.classifier = nn.Linear(self.base_model.fc.in_features, num_classes)
		self.dim = self.base_model.fc.in_features
		
	def forward(self, x):
		feature  = self.features(x)
		x = feature.view(feature.size(0), -1)		
		output = self.classifier(x)
		return output, x
	
	def get_embedding_dim(self):
		return self.dim




class MNIST_Net(nn.Module):
	def __init__(self, dim = 28 * 28, pretrained=False, num_classes = 10):
		super().__init__()
		resnet18 = models.resnet18(pretrained=pretrained)
		self.features = nn.Sequential(*list(resnet18.children())[:-1])
		self.conv = nn.Conv2d(1, 3, kernel_size = 1)
		self.conv1 = nn.Conv2d(1, 3, kernel_size=7, stride=2, padding=3, bias=False)
		self.classifier = nn.Linear(resnet18.fc.in_features,num_classes)
		self.dim = resnet18.fc.in_features
		
	
	def forward(self, x):
		x = self.conv(x)
		feature  = self.features(x)	
		x = feature.view(feature.size(0), -1)	
		output = self.classifier(x)
		return output, x
	
	def get_embedding_dim(self):
		return self.dim


