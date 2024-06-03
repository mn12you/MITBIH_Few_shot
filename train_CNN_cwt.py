import args as ar 
import os
import pandas as pd
import torch
import glob
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
import pickle
from itertools import product

from pathlib import Path

from tqdm import tqdm
import numpy as np
from model_code.relation_model import *
from data_mod.dataset import ECGDataset_pair, ECGDataset_few_shot, ECGDataset_all
from utils import *
from sklearn.metrics import confusion_matrix
from torchsummary import summary


def train(dataloader, net, arg, criterion, epoch, optimizer, device):
    print('Training epoch %d:' % epoch)
    net.train()
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data,labels) in enumerate(tqdm(dataloader)):
        data,labels= data.float().to(device), labels.float().to(device)
        optimizer.zero_grad()
        output=net(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    print(optimizer.param_groups[0]['lr'])
    loss_total=running_loss/(len(dataloader))
    print('Loss: %.4f' % loss_total)
    return loss_total

def evaluation(dataloader,net,arg,criterion,device):
    net.eval()
    flag=False
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data,labels) in enumerate(tqdm(dataloader)):
        data,labels= data.float().to(device), labels.float().to(device)
        output=net(data)
        loss = criterion(output, labels)
        running_loss += loss.item()
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    loss_total=running_loss/(len(dataloader))
    y_trues = np.vstack(labels_list)
    y_scores = np.vstack(output_list)
    
    if arg.phase=="Train":
        acc_val=cal_acc(y_trues,y_scores)
        print("Accuracy: %.3f, Loss: %3.f" %(acc_val,loss_total))
        if loss_total<arg.best_metric:
            arg.best_metric=loss_total
            arg.patience = 100
            torch.save(net.state_dict(), arg.model_path)
            print("Saved")
        else:
            arg.patience -= 1
            if arg.patience == 0:
                flag=True

    return y_trues,y_scores,loss_total,flag




def train_CNN(arg,name):

    arg.model_path=Path("./models",name,arg.model_name+".pth")
    arg.result_path=Path("./result",name)
    seed = arg.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if arg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = 'cpu'
    print(device)
    if arg.model_name=="SembedNet_2D":
        net=SembedNet_2D().to(device)
    # elif arg.model_name=="LMUEBCNet":
    #     net=LMUEBCNet().to(device)
    summary(net,(1,56,56))

    if arg.phase=="Train":
        train_data_path=Path("./data",name,"train","data",name+".npy")
        train_label_path=Path("./data",name,"train","label",name+".npy")
        val_data_path=Path("./data",name,"val","data",name+".npy")
        val_label_path=Path("./data",name,"val","label",name+".npy")
        train_dataset=ECGDataset_all(train_data_path,train_label_path)
        train_loader = DataLoader(train_dataset, batch_size=arg.batch_size, shuffle=True, num_workers=arg.num_workers, pin_memory=True)
        val_dataset=ECGDataset_all(val_data_path,val_label_path)
        val_loader = DataLoader(val_dataset, batch_size=arg.batch_size, shuffle=False, num_workers=arg.num_workers, pin_memory=True)
        optimizer = torch.optim.Adam(net.parameters(), lr=arg.lr)
        criterion = nn.CrossEntropyLoss()
        print("train")
        if arg.resume:
            net.load_state_dict(torch.load(arg.model_path,map_location=device))
        train_loss=[]
        val_loss=[]
        arg.best_metric=100.0
        arg.patience=100
        for epoch in range(arg.epochs):
            train_loss_temp=train(train_loader, net, arg, criterion, epoch, optimizer, device)
            train_loss.append(train_loss_temp)
            _,_,val_loss_temp,flag=evaluation(val_loader,net,arg,criterion,device)
            val_loss.append(val_loss_temp)
            if flag:
                break
        np.save(str(arg.result_path)+"/"+arg.model_name+"_train_loss.npy",train_loss)
        np.save(str(arg.result_path)+"/"+arg.model_name+"_val_loss.npy",val_loss)
    else:
        test_data_path=Path("./data",name,"test","data",name+".npy")
        test_label_path=Path("./data",name,"test","label",name+".npy")
        test_dataset=ECGDataset_all(test_data_path,test_label_path)
        test_loader = DataLoader(test_dataset, batch_size=arg.batch_size, shuffle=False, num_workers=arg.num_workers, pin_memory=True)
        criterion=nn.CrossEntropyLoss()
        net.load_state_dict(torch.load(arg.model_path, map_location=device))
        y_true,y_score,_,_=evaluation(test_loader,net,arg,criterion,device)
        result_path=Path(arg.result_path,arg.model_name)
        save_result(y_true,y_score,result_path,1)

def folder_generate(name):
    path=Path("./result",name)
    if not path.exists():
        os.makedirs(path)
    else:
         print("Dir exit.")
    path=Path("./models",name)
    if not path.exists():
        os.makedirs(path)
    else:
         print("Dir exit.")

def train_on_dataset(arg,dataset):
    for dataset_num in dataset:
        if arg.data_dir=="./mit_bih":
            name="mitbih_"+str(dataset_num)+"_cwt"
            folder_generate(name)
            train_CNN(arg,name)

if __name__=="__main__":
    arg = ar.parse_args()
    data_dir = os.path.normpath(arg.data_dir)
    database = os.path.basename(data_dir)
    dataset=[1,5,10,30,50,90,150]
    print(arg.data_dir)
    print("Train on:",arg.model_name)
    train_on_dataset(arg,dataset)
    arg.phase="test"
    train_on_dataset(arg,dataset)


