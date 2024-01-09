import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class Strategy:
    def __init__(self, dataset, net, args_input, args_task):
        self.dataset = dataset
        self.net = net
        self.args_input = args_input
        self.args_task = args_task

    def query(self, n):
        pass
    
    def get_labeled_count(self):
        labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        return len(labeled_idxs)
    
    def get_model(self):
        return self.net.get_model()

    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    def train(self, data = None, model_name = None):
        if model_name == None:
            if data == None:
                labeled_idxs, labeled_data = self.dataset.get_labeled_data()
                self.net.train(labeled_data)
            else:
                self.net.train(data)
        else:
            if model_name == 'WAAL':
                labeled_idxs, labeled_data = self.dataset.get_labeled_data()
                X_labeled, Y_labeled = self.dataset.get_partial_labeled_data()
                X_unlabeled, Y_unlabeled = self.dataset.get_partial_unlabeled_data()
                self.net.train(labeled_data, X_labeled, Y_labeled,X_unlabeled, Y_unlabeled)
            else:
                raise NotImplementedError

    def train_mae(self, data = None, unl_data = None, model_name = None):
        if model_name == None:
            if 'Sample' in self.args_input.aug_strategy:
                sample_aug_data_lst = self.dataset.get_labeled_aug_data_idxs()
            else:
                sample_aug_data_lst = None

            if data == None or unl_data == None:
                labeled_idxs, labeled_data = self.dataset.get_labeled_data()
                unl_idxs, unl_data = self.dataset.get_unlabeled_data()
                self.net.train(labeled_data,unl_data,sample_aug_data_lst=sample_aug_data_lst)
            else:
                self.net.train(data,unl_data,sample_aug_data_lst=sample_aug_data_lst)
        else:
            if model_name == 'WAAL':
                labeled_idxs, labeled_data = self.dataset.get_labeled_data()
                X_labeled, Y_labeled = self.dataset.get_partial_labeled_data()
                X_unlabeled, Y_unlabeled = self.dataset.get_partial_unlabeled_data()
                self.net.train(labeled_data, X_labeled, Y_labeled,X_unlabeled, Y_unlabeled)
            else:
                raise NotImplementedError
    
    def predict(self, data):
        preds = self.net.predict(data)
        return preds

    def predict_prob(self, data):
        probs = self.net.predict_prob(data,num_class = self.args_task['num_class'])
        return probs

    def predict_prob_dropout(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout(data, n_drop=n_drop,num_class = self.args_task['num_class'])
        return probs

    def predict_prob_dropout_split(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout_split(data, n_drop=n_drop,num_class = self.args_task['num_class'])
        return probs
    
    def get_embeddings(self, data):
        embeddings = self.net.get_embeddings(data)
        return embeddings
    
    def get_grad_embeddings(self, data):
        embeddings = self.net.get_grad_embeddings(data)
        return embeddings

