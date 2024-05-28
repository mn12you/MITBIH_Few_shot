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
from model_code.relation_model import transform_fully,SiameseNetwork,SembedNet
from data_mod.dataset import ECGDataset_pair, ECGDataset_few_shot, ECGDataset_all
# from dataset_alldata import ECGDataset_IE
from utils import *
from sklearn.metrics import confusion_matrix
from torchsummary import summary


# def evaluate_on_feature(dataloader,feature_net, net, args, criterion, device,prototype):
#     print('Validating...')
#     with open(args.multi_label_binarizer, 'rb') as tokenizer:
#         mlb=pickle.load(tokenizer)
#     feature_net.eval()
#     net.eval()
#     running_loss = 0
#     output_list, labels_list = [], []
#     for _, (data,labels) in enumerate(tqdm(dataloader)):
#         data,labels= data.float().to(device), labels.float().to(device)
#         prototype_embed=[]
#         for i in prototype.keys():
#             prototype_temp=feature_net(prototype[i])
#             prototype_embed.append(torch.mean(prototype_temp,dim=0).unsqueeze(0))
#         prototype_embed=torch.cat(prototype_embed,dim=0)
#         batch_embed=feature_net(data)
#         prototype_embed_ext=prototype_embed.unsqueeze(0).repeat(batch_embed.shape[0],1,1,1)
#         batch_embed_ext=batch_embed.unsqueeze(0).repeat(5,1,1,1)
#         batch_embed_ext = torch.transpose(batch_embed_ext,0,1)
#         relation_pairs = torch.cat((prototype_embed_ext,batch_embed_ext),2).view(-1,128*2,79)
#         output=net(relation_pairs).view(-1,5)
#         loss = criterion(output, labels)
#         running_loss += loss.item()
#         output_list.append(output.data.cpu().numpy())
#         labels_list.append(labels.data.cpu().numpy())



    # for _, (data,labels) in enumerate(tqdm(dataloader)):
    #     data,labels= data.float().to(device), labels.float().to(device)
    #     feature=feature_net(data)
    #     output_f=torch.tensor([]).to(device)
    #     for key in prototype.keys():
    #         output= net(feature)
    #         embed=torch.abs(feature-prototype[key])
    #         output=net(embed)
    #         output = torch.sigmoid(output)
    #         output_f=torch.cat((output_f,output),dim=1)
    #     loss = criterion(output_f, labels)
    #     running_loss += loss.item()
    #     output_list.append(output_f.data.cpu().numpy())
    #     labels_list.append(labels.data.cpu().numpy())


#     loss_total=running_loss/len(dataloader)
#     print('Loss: %.4f' % loss_total)
#     y_trues = np.vstack(labels_list)
#     y_scores = np.vstack(output_list)
#     auc=cal_auc(y_trues,y_scores)
#     if (args.phase=="train" and auc>args.best_metric):
#         args.best_metrics=auc
#         torch.save(net.state_dict(), args.model_path)
#         torch.save(feature_net.state_dict(), args.feature_model_path)
#         print("Saved")
    
#     print("Macro Precision: %.3f, Macro Recall: %.3f, Macro F1 score: %.3f, Macro AUC: %.3f, with threshold: %.3f" % cal_scores(y_trues,y_scores))
  
#     return y_trues,y_scores,loss_total

# def train_on_feature(dataloader, feature_net,net, args, criterion, epoch, scheduler, optimizer, device,prototype):
#     with open(args.multi_label_binarizer, 'rb') as tokenizer:
#         mlb=pickle.load(tokenizer)
#     print('Training epoch %d:' % epoch)
#     running_loss = 0
#     output_list, labels_list = [], []
#     for _, (data,labels) in enumerate(tqdm(dataloader)):
#         data,labels= data.float().to(device), labels.float().to(device)
#         prototype_embed=[]
#         for i in prototype.keys():
#             prototype_temp=feature_net(prototype[i])
#             prototype_embed.append(torch.mean(prototype_temp,dim=0).unsqueeze(0))
#         prototype_embed=torch.cat(prototype_embed,dim=0)
#         batch_embed=feature_net(data)
#         prototype_embed_ext=prototype_embed.unsqueeze(0).repeat(batch_embed.shape[0],1,1,1)
#         batch_embed_ext=batch_embed.unsqueeze(0).repeat(5,1,1,1)
#         batch_embed_ext = torch.transpose(batch_embed_ext,0,1)
#         relation_pairs = torch.cat((prototype_embed_ext,batch_embed_ext),2).view(-1,128*2,79)
#         output=net(relation_pairs).view(-1,5)
#         loss = criterion(output, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()

    # for _, (data,labels) in enumerate(tqdm(dataloader)):
    #     data,labels= data.float().to(device), labels.float().to(device)
    #     labels=torch.argmax(labels,axis=1)
    #     feature = feature_net(data)
    #     for key in prototype.keys():
    #         net.train()
    #         embed=torch.abs(feature-prototype[key])
    #         output=net(embed)
    #         output = torch.sigmoid(output)
    #         labels_single=(labels==key).float().unsqueeze(1)
    #         loss = criterion(output, labels_single)
    #         optimizer.zero_grad()
    #         loss.backward(retain_graph=True)
    #         optimizer.step()
    #         running_loss += loss.item()


    # print(optimizer.param_groups[0]['lr'])
    # scheduler.step()
    # loss_total=running_loss/(len(dataloader)*len(prototype.keys()))
    # print('Loss: %.4f' % loss_total)
    # return loss_total



# def train_on_feature_10fold(args,name,feature_model,shot):
    #         print("Evaluate")
    #         feature_net.load_state_dict(torch.load(args.feature_model_path,map_location=device))
    #         net.load_state_dict(torch.load(args.model_path, map_location=device))
    #         prototype={0:[],1:[],2:[],3:[],4:[]}
    #         for data, label in train_loader:
    #             data= data.float().to(device)
    #             # out=feature_net(data)
    #             for ind,labelty in enumerate(label):
    #                 prototype[np.where(labelty)[0][0]].append(data[ind].unsqueeze(0))
    #         for key in prototype.keys():
    #             prototype[key]=torch.cat(prototype[key], dim=0)
    #         y_true,y_score,loss_test=evaluate_on_feature(test_loader,feature_net, net, args, criterion, device,prototype)
    #         Final_output.append(y_score)
    #         Final_true.append(y_true)

    # if args.phase != 'train':
    #     Final_output=np.vstack(Final_output)
    #     Final_true=np.vstack(Final_true)
    #     print(Final_output.shape)
    #     print(Final_true.shape)
    #     path=(result_path+args.model_name+"_Final")
    #     save_result(Final_true,Final_output,path)
    
# def train_shot_on_feature(args,dataset,feature_model):
#     shots=[5,10,20,50]
#     for shot in shots:
#         train_on_feature_10fold(args,dataset,feature_model,shot)

# def folder_generate(name):
#     path=os.path.join("./models",name)
#     if not os.path.exists(path):
#         os.makedirs(path)

#     path=os.path.join("./result",name)
#     if not os.path.exists(path):
#         os.makedirs(path)


def train(dataloader, net, arg, criterion, epoch, optimizer, device):
    print('Training epoch %d:' % epoch)
    net.train()
    # encoder_net.train()
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
    # scheduler.step()
    loss_total=running_loss/(len(dataloader))
    print('Loss: %.4f' % loss_total)
    # y_trues = np.vstack(labels_list)
    # y_scores = np.vstack(output_list)
    # torch.save(encoder_net.state_dict(), arg.encoder_model_path)   
    # print("Macro Precision: %.3f, Macro Recall: %.3f, Macro F1 score: %.3f, Macro AUC: %.3f, with threshold: %.3f" % cal_scores(y_trues,y_scores))
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
        print(acc_val)
        if loss_total<arg.best_metric:
            arg.best_metric=loss_total
            arg.patience = 100
            torch.save(net.state_dict(), arg.encoder_model_path)
            print("Saved")
        else:
            arg.patience -= 1
            if arg.patience == 0:
                flag=True
        

    return y_trues,y_scores,loss_total,flag




def train_CNN(arg,name):
    if arg.phase=="Train":
        train_data_path=Path("./data",name,"train","data",name+".npy")
        train_label_path=Path("./data",name,"train","label",name+".npy")
        val_data_path=Path("./data",name,"val","data",name+".npy")
        val_label_path=Path("./data",name,"val","label",name+".npy")
        arg.encoder_model_path=Path("./models",name,"Sembed"+".pth")
        arg.result_path=Path("./result",name)
        train_dataset=ECGDataset_all(train_data_path,train_label_path)
        train_loader = DataLoader(train_dataset, batch_size=arg.batch_size, shuffle=True, num_workers=arg.num_workers, pin_memory=True)
        val_dataset=ECGDataset_all(val_data_path,val_label_path)
        val_loader = DataLoader(val_dataset, batch_size=arg.batch_size, shuffle=False, num_workers=arg.num_workers, pin_memory=True)
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
        # if arg.transform_model_name=="fully-connected":
        #     print("Train on:",arg.transform_model_name)
        #     net =transform_fully().to(device)
        # if arg.encoder_model_name=="Sembed":
        #     print("Output on:",arg.encoder_model_name)
        #     encoder_net =Sembedencoder().to(device)  
        print("Output on:","SiameseNetwork")
        net =SembedNet().to(device)  
        summary(net,(1,259))
        optimizer = torch.optim.Adam(net.parameters(), lr=arg.lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        print("train")
        if arg.resume:
            # net.load_state_dict(torch.load(arg.transform_model_path,map_location=device))
            # encoder_net.load_state_dict(torch.load(arg.encoder_model_path, map_location=device))
            net.load_state_dict(torch.load(arg.encoder_model_path,map_location=device))
        train_loss=[]
        val_loss=[]
        arg.best_metric=100.0
        arg.patience=100
        for epoch in range(arg.epochs):
            train_loss.append(train(train_loader, net, arg, criterion, epoch, optimizer, device))
            _,_,val_loss_temp,flag=evaluation(val_loader,net,arg,criterion,device)
            val_loss.append(val_loss_temp)
            if flag:
                break
        np.save(str(arg.result_path)+"/Sembed_train_loss.npy",train_loss)
        np.save(str(arg.result_path)+"/Sembed_val_loss.npy",val_loss)
    else:
        test_data_path=Path("./data",name,"test","data",name+".npy")
        test_label_path=Path("./data",name,"test","label",name+".npy")
        # arg.transform_model_path=Path("./models",name,arg.transform_model_name+".pth")
        # arg.encoder_model_path=Path("./models",name,arg.encoder_model_name+".pth")
        arg.encoder_model_path=Path("./models",name,"Sembed"+".pth")
        arg.result_path=Path("./result",name)
        test_dataset=ECGDataset_all(test_data_path,test_label_path)
        test_loader = DataLoader(test_dataset, batch_size=arg.batch_size, shuffle=False, num_workers=arg.num_workers, pin_memory=True)
        if arg.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = 'cpu'
        print(device)

            # if arg.transform_model_name=="fully-connected":
            #     print("Train on:",arg.transform_model_name)
            #     net =transform_fully().to(device)
            # if arg.encoder_model_name=="Sembed":
            #     print("Output on:",arg.encoder_model_name)
            #     encoder_net =Sembedencoder().to(device)  
        print("Output on:","Sembed")
        net =SembedNet().to(device)  
        summary(net,(1,259))
        criterion=nn.CrossEntropyLoss()
        # net.load_state_dict(torch.load(arg.transform_model_path,map_location=device))
        net.load_state_dict(torch.load(arg.encoder_model_path, map_location=device))
        y_true,y_score,_,_=evaluation(test_loader,net,arg,criterion,device)
        result_path=Path(arg.result_path,arg.encoder_model_name)
        save_result(y_true,y_score,result_path,1)
        

def train_on_dataset(arg,dataset):
    for dataset_num in dataset:
        if arg.data_dir=="./mit_bih":
            name="mitbih_"+str(dataset_num)
            train_CNN(arg,name)

if __name__=="__main__":
    arg = ar.parse_args()
    data_dir = os.path.normpath(arg.data_dir)
    database = os.path.basename(data_dir)
    dataset=[1,5,10,30,50,90,150]
    print(arg.data_dir)
    print("Train on:",arg.encoder_model_name)
    train_on_dataset(arg,dataset)
    arg.phase="test"
    train_on_dataset(arg,dataset)


