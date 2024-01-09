import numpy as np
import torch
import random
import os
from torchvision import datasets
from PIL import Image
from datasets_lib.medmnist import get_medmnist_data
from datasets_lib.pcam import PCAM
# from utils import get_dis_matrix
"""
Standard Data
"""

# Pipeline for Active Learning Data
class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, handler, args_task):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler
        self.args_task = args_task
        self.n_pool = len(X_train)
        self.n_test = len(X_test)
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        
        # distance matrix for sample aug data
        self.dis_mat = None
        self.dis_indices = None
    
    # Init Distance Matrix
    def get_dis_mat(self,args_input):
        # self.dis_mat = torch.load(os.path.join(args_input.dis_matrix_path,'dis_mat.pt'),map_location=torch.device('cpu'))
        self.dis_indices = torch.load(os.path.join(args_input.dis_matrix_path,'dis_indices.pt'),map_location=torch.device('cpu'))
        self.sample_top_k = args_input.sample_top_k
    
    # return a list where each item are the sample 
    def get_labeled_aug_data_idxs(self):
        labeled_idxs = self.get_labeled_data_idxs()
        unlabeled_idxs = self.get_unlabeled_data_idxs()
        labeled_y = self.Y_train[labeled_idxs]
        if self.dis_indices is None:
            raise ValueError('Get Distance Matrix First')
        else:
            sample_mat = self.dis_indices[labeled_idxs.tolist(),1:self.sample_top_k+1]
            aug_sample_data_lst = []
            for i in range(sample_mat.shape[0]):
                aug_sample_data_lst.append(self.handler(self.X_train[sample_mat[i]], self.Y_train[sample_mat[i]], self.args_task['transform_train']))
        return aug_sample_data_lst


        
    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True
    
    def get_unlabeled_data_by_idx(self, idx):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return self.X_train[unlabeled_idxs][idx]
    
    def get_data_by_idx(self, idx):
        return self.X_train[idx], self.Y_train[idx]

    def get_new_data(self, X, Y):
        return self.handler(X, Y, self.args_task['transform_train'])
    
    def get_labeled_data_idxs(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs
    
    def get_unlabeled_data_idxs(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs

    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs], self.args_task['transform_train'])
    
    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs], self.args_task['transform_train'])
    
    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train, self.args_task['transform_train'])

    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test, self.args_task['transform'])
    
    def get_partial_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return self.X_train[labeled_idxs], self.Y_train[labeled_idxs]
    
    def get_partial_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs]

    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_test==preds).sum().item() / self.n_test
    
    def cal_train_acc(self, preds, Y_train):
        return 1.0 * (self.Y_train==preds).sum().item() / self.n_test

def get_CIFAR10(handler, args_task, args):
    data_path = os.path.join(args.data_path,'CIFAR10')
    if os.path.exists(data_path) is False:
        os.makedirs(data_path)
    data_train = datasets.CIFAR10(data_path, train=True, download=True)
    data_test = datasets.CIFAR10(data_path, train=False, download=True)
    # import pdb; pdb.set_trace()
    return Data(data_train.data, torch.LongTensor(data_train.targets), data_test.data, torch.LongTensor(data_test.targets), handler, args_task)

def get_waterbirds(handler, args_task, args):
    import wilds
    from torchvision import transforms
    dataset = wilds.get_dataset(dataset='waterbirds', root_dir='./data/waterbirds', download='True')
    trans = transforms.Compose([transforms.Resize([255,255])])
    train = dataset.get_subset(split = 'train',transform = trans)
    test = dataset.get_subset(split = 'test', transform = trans)

    len_train = train.metadata_array.shape[0]
    len_test = test.metadata_array.shape[0]
    X_tr = []
    Y_tr = []
    X_te = []
    Y_te = []

    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    f = open('waterbirds.txt', 'w')

    for i in range(len_train):
        x,y,meta = train.__getitem__(i)
        img = np.array(x)
        X_tr.append(img)
        Y_tr.append(y)

    for i in range(len_test):
        x,y, meta = test.__getitem__(i)
        img = np.array(x)

        X_te.append(img)
        Y_te.append(y)
        if meta[0] == 0 and meta[1] == 0:
            f.writelines('1') #landbird_background:land
            f.writelines('\n')
            count1 = count1 + 1
        elif meta[0] == 1 and meta[1] == 0:
            f.writelines('2') #landbird_background:water
            count2 = count2 + 1
            f.writelines('\n')
        elif meta[0] == 0 and meta[1] == 1:
            f.writelines('3') #waterbird_background:land
            f.writelines('\n')
            count3 = count3 + 1
        elif meta[0] == 1 and meta[1] == 1:
            f.writelines('4') #waterbird_background:water
            f.writelines('\n')
            count4 = count4 + 1
        else:
            raise NotImplementedError    
    f.close()

    Y_tr = torch.tensor(Y_tr)
    Y_te = torch.tensor(Y_te)
    X_tr = np.array(X_tr)
    X_te = np.array(X_te)
    return Data(X_tr, Y_tr, X_te, Y_te, handler, args_task)





"""
Medical Data
"""
def get_PCAM(handler, args_task, args):
    data_path = os.path.join(args.data_path,'PCAM')
    if os.path.exists(data_path) is False:
        os.makedirs(data_path)
    raw_train = PCAM(
        root = data_path,
        split = 'train',
        download=True
    )
    raw_test =  PCAM(
        root = data_path,
        split = 'test',
        download=True
    )
    import pdb
    pdb.set_trace()




def get_DermaMNIST(handler, args_task, args):  
    data_path = os.path.join(args.data_path,'MedMNIST')
    if os.path.exists(data_path) is False:
        os.makedirs(data_path)
    _, raw_train = get_medmnist_data(data_name='dermamnist',data_path=data_path,split='train')
    _, raw_test = get_medmnist_data(data_name='dermamnist',data_path=data_path,split='test')
    return Data(raw_train.imgs, torch.LongTensor(raw_train.labels.ravel().tolist()), raw_test.imgs, torch.LongTensor(raw_test.labels.ravel().tolist()), handler, args_task)

def get_BreastMNIST(handler, args_task, args):  
    data_path = os.path.join(args.data_path,'MedMNIST')
    if os.path.exists(data_path) is False:
        os.makedirs(data_path)
    _, raw_train = get_medmnist_data(data_name='breastmnist',data_path=data_path,split='train')
    _, raw_test = get_medmnist_data(data_name='breastmnist',data_path=data_path,split='test')
    return Data(raw_train.imgs, torch.LongTensor(raw_train.labels.ravel().tolist()), raw_test.imgs, torch.LongTensor(raw_test.labels.ravel().tolist()), handler, args_task)

def get_BloodMNIST(handler, args_task, args):
    data_path = os.path.join(args.data_path,'MedMNIST')
    if os.path.exists(data_path) is False:
        os.makedirs(data_path)
    _, raw_train = get_medmnist_data(data_name='bloodmnist',data_path=data_path,split='train')
    _, raw_test = get_medmnist_data(data_name='bloodmnist',data_path=data_path,split='test')
    return Data(raw_train.imgs, torch.LongTensor(raw_train.labels.ravel().tolist()), raw_test.imgs, torch.LongTensor(raw_test.labels.ravel().tolist()), handler, args_task)


def get_PathMNIST(handler, args_task, args):  
    data_path = os.path.join(args.data_path,'MedMNIST')
    if os.path.exists(data_path) is False:
        os.makedirs(data_path)
    _, raw_train = get_medmnist_data(data_name='pathmnist',data_path=data_path,split='train')
    _, raw_test = get_medmnist_data(data_name='pathmnist',data_path=data_path,split='test')
    #PathMNIST has 9 classes
    return Data(raw_train.imgs, torch.LongTensor(raw_train.labels.ravel().tolist()), raw_test.imgs, torch.LongTensor(raw_test.labels.ravel().tolist()), handler, args_task)

def get_PneumoniaMNIST(handler, args_task, args):  
    data_path = os.path.join(args.data_path,'MedMNIST')
    if os.path.exists(data_path) is False:
        os.makedirs(data_path)
    _, raw_train = get_medmnist_data(data_name='pneumoniamnist',data_path=data_path,split='train')
    _, raw_test = get_medmnist_data(data_name='pneumoniamnist',data_path=data_path,split='test')
    return Data(raw_train.imgs, torch.LongTensor(raw_train.labels.ravel().tolist()), raw_test.imgs, torch.LongTensor(raw_test.labels.ravel().tolist()), handler, args_task)

def get_ChestMNIST(handler, args_task, args):  
    data_path = os.path.join(args.data_path,'MedMNIST')
    if os.path.exists(data_path) is False:
        os.makedirs(data_path)
    _, raw_train = get_medmnist_data(data_name='chestmnist',data_path=data_path,split='train')
    _, raw_test = get_medmnist_data(data_name='chestmnist',data_path=data_path,split='test')
    train_labels = [] 
    for i,y in enumerate(raw_train.labels):
        if np.all(y == 0):
            train_labels.append(0)
        else:
            train_labels.append(1)
    
    test_labels = [] 
    for i,y in enumerate(raw_test.labels):
        if np.all(y == 0):
            test_labels.append(0)
        else:
            test_labels.append(1)
    return Data(raw_train.imgs, torch.LongTensor(train_labels), raw_test.imgs, torch.LongTensor(test_labels), handler, args_task)

def get_TissueMNIST(handler, args_task, args):  
    data_path = os.path.join(args.data_path,'MedMNIST')
    if os.path.exists(data_path) is False:
        os.makedirs(data_path)
    _, raw_train = get_medmnist_data(data_name='tissuemnist',data_path=data_path,split='train')
    _, raw_test = get_medmnist_data(data_name='tissuemnist',data_path=data_path,split='test')
    #TissueMNIST has 8 classes
    return Data(raw_train.imgs, torch.LongTensor(raw_train.labels.ravel().tolist()), raw_test.imgs, torch.LongTensor(raw_test.labels.ravel().tolist()), handler, args_task)

def get_OCTMNIST(handler, args_task, args):  
    data_path = os.path.join(args.data_path,'MedMNIST')
    if os.path.exists(data_path) is False:
        os.makedirs(data_path)
    _, raw_train = get_medmnist_data(data_name='octmnist',data_path=data_path,split='train')
    _, raw_test = get_medmnist_data(data_name='octmnist',data_path=data_path,split='test')
    #OCTMNIST has 4 classes
    return Data(raw_train.imgs, torch.LongTensor(raw_train.labels.ravel().tolist()), raw_test.imgs, torch.LongTensor(raw_test.labels.ravel().tolist()), handler, args_task)

def get_OrganAMNIST(handler, args_task, args):  
    data_path = os.path.join(args.data_path,'MedMNIST')
    if os.path.exists(data_path) is False:
        os.makedirs(data_path)
    _, raw_train = get_medmnist_data(data_name='organamnist',data_path=data_path,split='train')
    _, raw_test = get_medmnist_data(data_name='organamnist',data_path=data_path,split='test')
    #OrganAMNIST has 11 classes
    return Data(raw_train.imgs, torch.LongTensor(raw_train.labels.ravel().tolist()), raw_test.imgs, torch.LongTensor(raw_test.labels.ravel().tolist()), handler, args_task)

def get_OrganCMNIST(handler, args_task, args):  
    data_path = os.path.join(args.data_path,'MedMNIST')
    if os.path.exists(data_path) is False:
        os.makedirs(data_path)
    _, raw_train = get_medmnist_data(data_name='organcmnist',data_path=data_path,split='train')
    _, raw_test = get_medmnist_data(data_name='organcmnist',data_path=data_path,split='test')
    #OrganCMNIST has 11 classes
    return Data(raw_train.imgs, torch.LongTensor(raw_train.labels.ravel().tolist()), raw_test.imgs, torch.LongTensor(raw_test.labels.ravel().tolist()), handler, args_task)

def get_OrganSMNIST(handler, args_task, args):  
    data_path = os.path.join(args.data_path,'MedMNIST')
    if os.path.exists(data_path) is False:
        os.makedirs(data_path)
    _, raw_train = get_medmnist_data(data_name='organsmnist',data_path=data_path,split='train')
    _, raw_test = get_medmnist_data(data_name='organsmnist',data_path=data_path,split='test')
    #OrganSMNIST has 11 classes
    return Data(raw_train.imgs, torch.LongTensor(raw_train.labels.ravel().tolist()), raw_test.imgs, torch.LongTensor(raw_test.labels.ravel().tolist()), handler, args_task)   






