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
from data_mod.dataset import ECGDataset_pair, ECGDataset_few_shot
from utils import *
from sklearn.metrics import confusion_matrix
from torchsummary import summary

def train(dataloader, net, arg, criterion, epoch, optimizer, device):
    print('Training epoch %d:' % epoch)
    net.train()
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data1,data2,labels) in enumerate(tqdm(dataloader)):
        data1,data2,labels= data1.float().to(device), data2.float().to(device), labels.float().to(device)
        optimizer.zero_grad()
        output=net(data1,data2)
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

def evaluation(dataloader,net,arg,criterion,shot,device):
    net.eval()
    flag=False
    running_loss = 0
    output_list, labels_list = [], []
    for _, (support,query,labels) in enumerate(tqdm(dataloader)):
        support,query,labels= support.float().to(device), query.float().to(device), labels.float().to(device)
        batch=labels.shape[0]
        class_num=labels.shape[-1]
        support=support.view(batch*class_num*shot,56,56).unsqueeze(1)
        query=query.unsqueeze(1).repeat(1,class_num*shot,1,1,1).view(batch*class_num*shot,1,56,56)
        output=net(query,support)
        if shot>1:
            output=output.view(batch,shot,class_num,-1)
            output=output.sum(dim=1).squeeze(-1)
        else:
            output=output.view(-1,class_num)
        if shot==1:   
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




def train_relation(arg,name):

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
    print("Output on:",arg.model_name)
    if arg.model_name=="Siamese_Sembed_2D":
        net=Siamese_Sembed_2D().to(device)
    # elif arg.model_name=="Siamese_LMU":
    #     net=Siamese_LMU().to(device)
    summary(net,[(1,56,56),(1,56,56)])


    if arg.phase=="Train":
        data_path_1=Path("./data",name,"train","data",name+"1.npy")
        data_path_2=Path("./data",name,"train","data",name+"2.npy")
        label_path=Path("./data",name,"train","label",name+".npy")
        support_path=Path("./data",name,"val","data",name+"_support_1_shot.npy")
        query_path=Path("./data",name,"val","data",name+"_query_1_shot.npy")
        label_val_path=Path("./data",name,"val","label",name+"_1_shot.npy")
        train_dataset=ECGDataset_pair(data_path_1,data_path_2,label_path)
        train_loader = DataLoader(train_dataset, batch_size=arg.batch_size, shuffle=True, num_workers=arg.num_workers, pin_memory=True)
        val_dataset=ECGDataset_few_shot(support_path,query_path,label_val_path)
        val_loader = DataLoader(val_dataset, batch_size=arg.batch_size, shuffle=False, num_workers=arg.num_workers, pin_memory=True)
        optimizer = torch.optim.Adam(net.parameters(), lr=arg.lr)
        criterion = nn.BCELoss()
        print("train")
        if arg.resume:
            net.load_state_dict(torch.load(arg.model_path,map_location=device))
        train_loss=[]
        val_loss=[]
        arg.best_metric=100.0
        arg.patience=100
        shot=1
        for epoch in range(arg.epochs):
            train_loss_temp=train(train_loader, net, arg, criterion, epoch, optimizer, device)
            train_loss.append(train_loss_temp)
            _,_,val_loss_temp,flag=evaluation(val_loader,net,arg,criterion,shot,device)
            val_loss.append(val_loss_temp)
            if flag:
                break
        np.save(str(arg.result_path)+"/"+arg.model_name+"_train_loss.npy",train_loss)
        np.save(str(arg.result_path)+"/"+arg.model_name+"_val_loss.npy",val_loss)
    else:
        shots=[1,5]
        for shot in shots:
            support_path=Path("./data",name,"test","data",name+"_support_"+str(shot)+"_shot.npy")
            query_path=Path("./data",name,"test","data",name+"_query_"+str(shot)+"_shot.npy")
            label_path=Path("./data",name,"test","label",name+"_"+str(shot)+"_shot.npy")
            test_dataset=ECGDataset_few_shot(support_path,query_path,label_path)
            test_loader = DataLoader(test_dataset, batch_size=arg.batch_size, shuffle=False, num_workers=arg.num_workers, pin_memory=True)
            criterion = nn.BCELoss()
            net.load_state_dict(torch.load(arg.model_path, map_location=device))
            y_true,y_score,_,_=evaluation(test_loader,net,arg,criterion,shot,device)
            result_path=Path(arg.result_path,arg.model_name)
            save_result(y_true,y_score,result_path,shot)

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
            name="mitbih_"+str(dataset_num)+"_cwt_pair"
            folder_generate(name)
            train_relation(arg,name)

if __name__=="__main__":
    arg = ar.parse_args()
    data_dir = os.path.normpath(arg.data_dir)
    database = os.path.basename(data_dir)
    dataset=[150]
    print(arg.data_dir)
    print("Train on:",arg.model_name)
    train_on_dataset(arg,dataset)
    arg.phase="test"
    train_on_dataset(arg,dataset)
    