import sys
import os 
from pathlib import Path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from args import parse_args
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt
from data_mod.dataset import ECGDataset
from sklearn.model_selection import train_test_split
import random


def folder_generate(name):
    data_diff=["10","50","90","150","500"]

    for diff in data_diff:
        path=Path("./data",name+"_"+diff)
        if not path.exists():
            path.mkdir()
            train_path=Path(path,"train")
            train_path.mkdir()
            train_path_sub=Path(train_path,"data")
            train_path_sub.mkdir()
            train_path_sub=Path(train_path,"label")
            train_path_sub.mkdir()
            test_path=Path(path,"test")
            test_path.mkdir()
            test_path_sub=Path(test_path,"data")
            test_path_sub.mkdir()
            test_path_sub=Path(test_path,"label")
            test_path_sub.mkdir()
        else:
            print("Dir exit.")
        path=Path("./result",name+"_"+diff)
        if not path.exists():
            os.makedirs(path)
        else:
            print("Dir exit.")



    
if __name__=="__main__":
    arg=parse_args()
    datadir=arg.data_dir
    basepath="mitbih"
    random.seed(arg.seed)
    folder_generate(basepath)
    train_dataset = ECGDataset()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True,prefetch_factor=3)
    output_list=[]
    labels_list=[]
    for _, (data, labels) in enumerate(tqdm(train_loader)):
        output_list.append(data)
        labels_list.append(labels)
    y_data = np.vstack(output_list)
    y_label = np.vstack(labels_list)
    y_data=np.expand_dims(y_data, axis=1)
    y_label=np.squeeze(y_label,axis=None)
    print(y_label.shape)
    print(y_data.shape)
    data_diff=["150","500"]
    for diff in data_diff:
        base_path=path=Path("./data",basepath+"_"+diff)
        train_data_path=Path(base_path,"train","data",basepath+"_"+diff+".npy")
        train_label_path=Path(base_path,"train","label",basepath+"_"+diff+".npy")
        test_data_path=Path(base_path,"test","data",basepath+"_"+diff+".npy")
        test_label_path=Path(base_path,"test","label",basepath+"_"+diff+".npy")
        random_num=int(diff)
        train_data=[]
        train_label=[]
        for class_num in range(y_label.shape[-1]):
            c_index=frozenset(np.where(y_label[:,class_num]==1)[0].tolist())
            shot_temp= random.sample(c_index, random_num)
            train_data=train_data+shot_temp
            train_label=train_label+shot_temp
        data=y_data[train_data]
        label=y_label[train_label]
        print(diff)
        print(data.shape)
        print(label.shape)
        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.4, random_state=arg.seed)
        np.save(train_data_path,X_train)
        np.save(train_label_path,y_train)
        np.save(test_data_path,X_test)
        np.save(test_label_path,y_test)
            



            