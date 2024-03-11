from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import os
import pandas as pd
import torch
import numpy as np

class Googdata(Dataset):
    def __init__(self, split, data_path,label="GOOG"):
        r"""
        Init method for initializing the dataset properties
        :param split: train/test to locate the image files
        :param im_path: root folder of images
        :param im_ext: image extension. assumes all
        images would be this type.
        """
        self.split = split
        self.data = self.load_data(data_path,label)
        self.label = label

    def load_data(self, data_path,label):
        if not os.path.isfile(data_path):
            assert os.path.exists(f"data/{str(label)}.csv"), "raw data {} does not exist".format(label)
            data = pd.read_csv(f"data/{str(label)}.csv")
            train_data = data[:int(len(data)*0.7)]
            test_data = data[int(len(data)*0.7):]
            # print(train_data)
            
            directory_train = f"data/train_{str(label)}"
            directory_test = f"data/test_{str(label)}"
            if not os.path.exists(directory_train):
                os.makedirs(directory_train)
            if not os.path.exists(directory_test):
                os.makedirs(directory_test)
            train_num = int(len(train_data)/(28*28))
            test_num = int(len(test_data)/(28*28))
            for i in range(train_num):
                train_part = train_data[i*28*28:(i+1)*28*28].values.reshape((28,28))
                np.savetxt(f"data/train_{str(label)}/{str(label)}_{i}.csv",train_part, delimiter=",")
            for i in range(test_num):
                test_part = test_data[i*28*28:(i+1)*28*28].values.reshape((28,28))
                np.savetxt(f"data/test_{str(label)}/{str(label)}_{i}.csv",test_part, delimiter=",")
  
        assert os.path.exists(data_path), "data path {} does not exist".format(data_path)
        datasets = []
        for fname in os.listdir(data_path):
            datasets.append(f"{data_path}/{fname}")
        return datasets
        
        
    def __len__(self):
        return len(self.data)   

    def __getitem__(self, index):
        data = pd.read_csv(self.data[index], index_col=False,header=None)
        data_tensor = torch.tensor(data.values.reshape((1,28,28)))
    
        
        # Convert input to -1 to 1 range.
        data_tensor = data_tensor.div_(torch.norm(data_tensor,2))
        return data_tensor 




