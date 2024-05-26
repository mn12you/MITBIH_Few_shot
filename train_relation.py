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
from model_code.relation_model import transform_fully, Sembedencoder
from data_mod.dataset import ECGDataset_pair, ECGDataset_few_shot
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


def train(dataloader,encoder_net, net, arg, criterion, epoch, optimizer, device):
    print('Training epoch %d:' % epoch)
    net.train()
    encoder_net.train()
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data1,data2,labels) in enumerate(tqdm(dataloader)):
        data1,data2,labels= data1.float().to(device), data2.float().to(device), labels.float().to(device)
        embed1=encoder_net(data1)
        embed2=encoder_net(data2)
        distance=torch.norm((embed1-embed2), p=1, dim=1)
        distance=torch.unsqueeze(distance, 1)
        m=nn.Sigmoid()
        output=m(net(distance))
        loss = criterion(output, labels)
        optimizer.zero_grad()
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
    torch.save(net.state_dict(), arg.transform_model_path)
    torch.save(encoder_net.state_dict(), arg.encoder_model_path)   
    # print("Macro Precision: %.3f, Macro Recall: %.3f, Macro F1 score: %.3f, Macro AUC: %.3f, with threshold: %.3f" % cal_scores(y_trues,y_scores))
    return loss_total

def evaluation(dataloader,encoder_net,net,arg,shot,device):
    net.eval()
    encoder_net.eval()
    running_loss = 0
    output_list, labels_list = [], []
    for _, (support,query,labels) in enumerate(tqdm(dataloader)):
        support,query,labels= support.float().to(device), query.float().to(device), labels.float().to(device)
        batch=labels.shape[0]
        class_num=labels.shape[-1]
        support=support.view(batch*class_num*shot,-1).unsqueeze(1)
        embed_query=encoder_net(query)
        embed_support=encoder_net(support)
        embed_query_ext=embed_query.unsqueeze(1).repeat(1,class_num*shot,1)
        embed_query_ext=embed_query_ext.view(batch*class_num*shot,-1)
        distance=torch.norm((embed_query_ext-embed_support), p=1, dim=1)
        distance=torch.unsqueeze(distance, 1)
        m=nn.Sigmoid()
        if shot>1:
            output=m(net(distance)).view(batch,shot,class_num,-1)
            output=output.sum(dim=1).squeeze(-1)
        else:
            output=m(net(distance)).view(-1,class_num)
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    y_trues = np.vstack(labels_list)
    y_scores = np.vstack(output_list)
    return y_trues,y_scores




def train_relation(arg,name):
    if arg.phase=="Train":
        data_path_1=Path("./data",name,"train","data",name+"1.npy")
        data_path_2=Path("./data",name,"train","data",name+"2.npy")
        label_path=Path("./data",name,"train","label",name+".npy")
        arg.transform_model_path=Path("./models",name,arg.transform_model_name+".pth")
        arg.encoder_model_path=Path("./models",name,arg.encoder_model_name+".pth")
        arg.result_path=Path("./result",name)
        train_dataset=ECGDataset_pair(data_path_1,data_path_2,label_path)
        train_loader = DataLoader(train_dataset, batch_size=arg.batch_size, shuffle=True, num_workers=arg.num_workers, pin_memory=True)
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
        if arg.transform_model_name=="fully-connected":
            print("Train on:",arg.transform_model_name)
            net =transform_fully().to(device)
        if arg.encoder_model_name=="Sembed":
            print("Output on:",arg.encoder_model_name)
            encoder_net =Sembedencoder().to(device)  
        summary(net,(1,))
        summary(encoder_net,(1,259))
        optimizer = torch.optim.Adam(net.parameters(), lr=arg.lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
        criterion = nn.BCELoss()
        print("train")
        if arg.resume:
            net.load_state_dict(torch.load(arg.transform_model_path,map_location=device))
            encoder_net.load_state_dict(torch.load(arg.encoder_model_path, map_location=device))
        train_loss=[]
        for epoch in range(arg.epochs):
            train_loss.append(train(train_loader,encoder_net, net, arg, criterion, epoch, optimizer, device))
        np.save(arg.result_path+"train_loss.npy",train_loss)
    else:
        shots=[1,5]
        for shot in shots:
            support_path=Path("./data",name,"test","data",name+"_support_"+str(shot)+"_shot.npy")
            query_path=Path("./data",name,"test","data",name+"_query_"+str(shot)+"_shot.npy")
            label_path=Path("./data",name,"test","label",name+"_"+str(shot)+"_shot.npy")
            arg.transform_model_path=Path("./models",name,arg.transform_model_name+".pth")
            arg.encoder_model_path=Path("./models",name,arg.encoder_model_name+".pth")
            arg.result_path=Path("./result",name)
            test_dataset=ECGDataset_few_shot(support_path,query_path,label_path)
            test_loader = DataLoader(test_dataset, batch_size=arg.batch_size, shuffle=True, num_workers=arg.num_workers, pin_memory=True)
            if arg.use_gpu and torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = 'cpu'
            print(device)

            if arg.transform_model_name=="fully-connected":
                print("Train on:",arg.transform_model_name)
                net =transform_fully().to(device)
            if arg.encoder_model_name=="Sembed":
                print("Output on:",arg.encoder_model_name)
                encoder_net =Sembedencoder().to(device)  
            summary(net,(1,))
            summary(encoder_net,(1,259))
            net.load_state_dict(torch.load(arg.transform_model_path,map_location=device))
            encoder_net.load_state_dict(torch.load(arg.encoder_model_path, map_location=device))
            y_true,y_score=evaluation(test_loader,encoder_net,net,arg,shot,device)
            result_path=Path(arg.result_path,arg.encoder_model_name)
            save_result(y_true,y_score,result_path,shot)
        

def train_on_dataset(arg,dataset):
    for dataset_num in dataset:
        if arg.data_dir=="./mit_bih":
            name="mitbih_"+str(dataset_num)+"_pair"
            train_relation(arg,name)

if __name__=="__main__":
    arg = ar.parse_args()
    data_dir = os.path.normpath(arg.data_dir)
    database = os.path.basename(data_dir)
    dataset=[10]
    print(arg.data_dir)
    print("Train on:",arg.encoder_model_name)
    print("Train on:",arg.transform_model_name)
    train_on_dataset(arg,dataset)
