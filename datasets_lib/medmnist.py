import medmnist
from medmnist import INFO

def get_medmnist_data(data_name,data_path,split='train'):
    download = True
    info = INFO[data_name]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])
    dataset = DataClass(split=split, download=download, root=data_path)
    data_info = {'task':task,'n_channels':n_channels,'n_classes':n_classes}
    return data_info,dataset

