import os
from torchvision import transforms
import random

SEED = 4666


args_pool = {
			'CIFAR10':
				{'n_epoch': 40,   
				 'name': 'CIFAR10',
				 'transform_train': transforms.Compose([
					transforms.Resize(224),
					transforms.ToTensor(),
				 	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
				 'transform': transforms.Compose([transforms.Resize(224),transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
				 'loader_tr_args':{'batch_size': 32, 'num_workers': 4},
				 'loader_te_args':{'batch_size': 256, 'num_workers': 4},
				 'num_class':10,
				 'pretrained': False,
				 'optimizer':'Adam',
				 'optimizer_args':{'lr': 0.001},
				 'img_size':224,
                 'mean':(0.4914, 0.4822, 0.4465),
				 'std':(0.2470, 0.2435, 0.2616)
				 },
			'PneumoniaMNIST':
				{'n_epoch': 10,    
				 'name': 'PneumoniaMNIST',
				 'transform_train':transforms.Compose([transforms.Resize(255),
                    	transforms.CenterCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(10),
                        transforms.RandomGrayscale(),
                        transforms.RandomAffine(translate=(0.05,0.05), degrees=0),
                        transforms.ToTensor()]),
				 'transform':transforms.Compose([transforms.Resize(255),
                        transforms.CenterCrop(224),
						transforms.ToTensor()]),
				 'loader_tr_args':{'batch_size': 32, 'num_workers': 2},
				 'loader_te_args':{'batch_size': 128, 'num_workers': 2},
				 'num_class':2,
				 'optimizer':'SGD',
				 'pretrained': False,
				 'optimizer_args':{'lr': 0.001}},
			'DermaMNIST':
				{'n_epoch': 80,
				  'name': 'DermaMNIST',
				  'transform_train': transforms.Compose([
					transforms.Resize(224),
					transforms.ToTensor(),
					transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
				  ]),
				  'transform': transforms.Compose([
					transforms.Resize(224),
					transforms.ToTensor(),
					transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
				  ]),
				  'loader_tr_args':{'batch_size': 32, 'num_workers': 2},
				  'loader_te_args':{'batch_size': 128, 'num_workers': 2},
				  'optimizer':'Adam',
				  'pretrained':False,
				  'num_class':7,
				  'optimizer_args':{'lr': 0.01},
				  'img_size':224,
				  'mean':(0.5,0.5,0.5),
				  'std':(0.5,0.5,0.5)
				},
			'DermaMNIST_Pretrain':
				{'n_epoch': 80,
				  'name': 'DermaMNIST',
				  'transform_train': transforms.Compose([
					transforms.Resize(224),
					transforms.ToTensor(),
					transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
				  ]),
				  'transform': transforms.Compose([
					transforms.Resize(224),
					transforms.ToTensor(),
					transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
				  ]),
				  'loader_tr_args':{'batch_size': 32, 'num_workers': 2},
				  'loader_te_args':{'batch_size': 128, 'num_workers': 2},
				  'optimizer':'Adam',
				  'pretrained':False,
				  'num_class':7,
				  'optimizer_args':{'lr': 0.01},
				  'mean':(0.5,0.5,0.5),
				  'std':(0.5,0.5,0.5)
				},
            'BloodMNIST':
				{'n_epoch': 80,
				  'name': 'BloodMNIST',
				  'transform_train': transforms.Compose([
					transforms.Resize(224),
					transforms.Grayscale(num_output_channels=3),
					transforms.ToTensor(),
					transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
				  ]),
				  'transform': transforms.Compose([
					transforms.Resize(224),
					transforms.Grayscale(num_output_channels=3),
					transforms.ToTensor(),
					transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
				  ]),
				  'loader_tr_args':{'batch_size': 32, 'num_workers': 2},
				  'loader_te_args':{'batch_size': 128, 'num_workers': 2},
				  'optimizer':'Adam',
				  'pretrained':False,
				  'num_class':8,
				  'optimizer_args':{'lr': 0.01},
                  'img_size':224,
                  'is_gray':True,
				  'mean':(0.5,0.5,0.5),
				  'std':(0.5,0.5,0.5)
			    },
            'BreastMNIST':
				{'n_epoch': 80,
				  'name': 'BreastMNIST',
				  'transform_train': transforms.Compose([
					transforms.Resize(224),
					transforms.Grayscale(num_output_channels=3),
					transforms.ToTensor(),
					transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
				  ]),
				  'transform': transforms.Compose([
					transforms.Resize(224),
					transforms.Grayscale(num_output_channels=3),
					transforms.ToTensor(),
					transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
				  ]),
				  'loader_tr_args':{'batch_size': 32, 'num_workers': 2},
				  'loader_te_args':{'batch_size': 128, 'num_workers': 2},
				  'optimizer':'Adam',
				  'pretrained':False,
				  'num_class':2,
				  'optimizer_args':{'lr': 0.01},
                  'img_size':224,
                  'is_gray':True,
				  'mean':(0.5,0.5,0.5),
				  'std':(0.5,0.5,0.5)
			    },
            'PathMNIST':
				{'n_epoch': 80,
				  'name': 'PathMNIST',
				  'transform_train': transforms.Compose([
					transforms.Resize(224),
					transforms.ToTensor(),
					transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
				  ]),
				  'transform': transforms.Compose([
					transforms.Resize(224),
					transforms.ToTensor(),
					transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
				  ]),
				  'loader_tr_args':{'batch_size': 32, 'num_workers': 2},
				  'loader_te_args':{'batch_size': 128, 'num_workers': 2},
				  'optimizer':'Adam',
				  'pretrained':False,
				  'num_class':9,
				  'optimizer_args':{'lr': 0.01},
                  'img_size':224,
				  'mean':(0.5,0.5,0.5),
				  'std':(0.5,0.5,0.5)
			    },
            'PneumoniaMNIST':
				{'n_epoch': 80,
				  'name': 'PneumoniaMNIST',
				  'transform_train': transforms.Compose([
					transforms.Resize(224),
					transforms.Grayscale(num_output_channels=3),
					transforms.ToTensor(),
					transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
				  ]),
				  'transform': transforms.Compose([
					transforms.Resize(224),
					transforms.Grayscale(num_output_channels=3),
					transforms.ToTensor(),
					transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
				  ]),
				  'loader_tr_args':{'batch_size': 32, 'num_workers': 2},
				  'loader_te_args':{'batch_size': 128, 'num_workers': 2},
				  'optimizer':'Adam',
				  'pretrained':False,
				  'num_class':2,
				  'optimizer_args':{'lr': 0.01},
                  'img_size':224,
				  'is_gray':True,
				  'mean':(0.5,0.5,0.5),
				  'std':(0.5,0.5,0.5)
			    },
            'ChestMNIST':
				{'n_epoch': 80,
				  'name': 'ChestMNIST',
				  'transform_train': transforms.Compose([
					transforms.Resize(224),
					transforms.Grayscale(num_output_channels=3),
					transforms.ToTensor(),
					transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
				  ]),
				  'transform': transforms.Compose([
					transforms.Resize(224),
					transforms.Grayscale(num_output_channels=3),
					transforms.ToTensor(),
					transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
				  ]),
				  'loader_tr_args':{'batch_size': 32, 'num_workers': 2},
				  'loader_te_args':{'batch_size': 128, 'num_workers': 2},
				  'optimizer':'Adam',
				  'pretrained':False,
				  'num_class':2,
				  'optimizer_args':{'lr': 0.01}
			    },
            'TissueMNIST':
				{'n_epoch': 80,
				  'name': 'TissueMNIST',
				  'transform_train': transforms.Compose([
					transforms.Resize(224),
					transforms.Grayscale(num_output_channels=3),
					transforms.ToTensor(),
					transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
				  ]),
				  'transform': transforms.Compose([
					transforms.Resize(224),
					transforms.Grayscale(num_output_channels=3),
					transforms.ToTensor(),
					transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
				  ]),
				  'loader_tr_args':{'batch_size': 32, 'num_workers': 2},
				  'loader_te_args':{'batch_size': 128, 'num_workers': 2},
				  'optimizer':'Adam',
				  'pretrained':False,
				  'num_class':8,
				  'optimizer_args':{'lr': 0.01},
                  'img_size':224,
				  'mean':(0.5,0.5,0.5),
				  'std':(0.5,0.5,0.5)
			    },
            'OCTMNIST':
				{'n_epoch': 80,
				  'name': 'OCTMNIST',
				  'transform_train': transforms.Compose([
					transforms.Resize(224),
					transforms.Grayscale(num_output_channels=3),
					transforms.ToTensor(),
					transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
				  ]),
				  'transform': transforms.Compose([
					transforms.Resize(224),
					transforms.Grayscale(num_output_channels=3),
					transforms.ToTensor(),
					transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
				  ]),
				  'loader_tr_args':{'batch_size': 32, 'num_workers': 2},
				  'loader_te_args':{'batch_size': 128, 'num_workers': 2},
				  'optimizer':'Adam',
				  'pretrained':False,
				  'num_class':4,
				  'optimizer_args':{'lr': 0.01}
			    },
            'OrganAMNIST':
				{'n_epoch': 80,
				  'name': 'OrganAMNIST',
				  'transform_train': transforms.Compose([
					transforms.Resize(224),
					transforms.Grayscale(num_output_channels=3),
					transforms.ToTensor(),
					transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
				  ]),
				  'transform': transforms.Compose([
					transforms.Resize(224),
					transforms.Grayscale(num_output_channels=3),
					transforms.ToTensor(),
					transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
				  ]),
				  'loader_tr_args':{'batch_size': 32, 'num_workers': 2},
				  'loader_te_args':{'batch_size': 128, 'num_workers': 2},
				  'optimizer':'Adam',
				  'pretrained':False,
				  'num_class':11,
				  'optimizer_args':{'lr': 0.01},
                  'img_size':224,
                  'is_gray':True,
				  'mean':(0.5,0.5,0.5),
				  'std':(0.5,0.5,0.5)
			    },
            'OrganCMNIST':
				{'n_epoch': 80,
				  'name': 'OrganCMNIST',
				  'transform_train': transforms.Compose([
					transforms.Resize(224),
					transforms.Grayscale(num_output_channels=3),
					transforms.ToTensor(),
					transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
				  ]),
				  'transform': transforms.Compose([
					transforms.Resize(224),
					transforms.Grayscale(num_output_channels=3),
					transforms.ToTensor(),
					transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
				  ]),
				  'loader_tr_args':{'batch_size': 32, 'num_workers': 2},
				  'loader_te_args':{'batch_size': 128, 'num_workers': 2},
				  'optimizer':'Adam',
				  'pretrained':False,
				  'num_class':11,
				  'optimizer_args':{'lr': 0.01},
                  'img_size':224,
                  'is_gray':True,
				  'mean':(0.5,0.5,0.5),
				  'std':(0.5,0.5,0.5)
			    },
            'OrganSMNIST':
				{'n_epoch': 80,
				  'name': 'OrganSMNIST',
				  'transform_train': transforms.Compose([
					transforms.Resize(224),
					transforms.Grayscale(num_output_channels=3),
					transforms.ToTensor(),
					transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
				  ]),
				  'transform': transforms.Compose([
					transforms.Resize(224),
					transforms.Grayscale(num_output_channels=3),
					transforms.ToTensor(),
					transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
				  ]),
				  'loader_tr_args':{'batch_size': 32, 'num_workers': 2},
				  'loader_te_args':{'batch_size': 128, 'num_workers': 2},
				  'optimizer':'Adam',
				  'pretrained':False,
				  'num_class':11,
				  'optimizer_args':{'lr': 0.01},
                  'img_size':224,
                  'is_gray':True,
				  'mean':(0.5,0.5,0.5),
				  'std':(0.5,0.5,0.5)
			    },
            
			}



