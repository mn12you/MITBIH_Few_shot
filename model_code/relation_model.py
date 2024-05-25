import torch
import torch.nn as nn
import os

class RelationNet(nn.Sequential):
     def __init__(self,kernel_size=5,num_classes=5,input_channels=128,inplanes=128, stride=2):
        layer_temp=[]
        input_channels=inplanes*2
        for i in range(2):
            if i>0:
                input_channels=input_channels//2
            layer_temp.append(nn.Conv1d(input_channels, inplanes,kernel_size=kernel_size,
                                        stride=stride, padding=(kernel_size-1)//2,bias=False))
            layer_temp.append(nn.BatchNorm1d(inplanes))
            layer_temp.append(nn.ReLU(inplace=True))
            layer_temp.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        layer_temp.append(nn.Flatten())
        layer_temp.append(nn.Linear(input_channels*5,128))
        layer_temp.append(nn.ReLU(inplace=True))
        layer_temp.append(nn.Linear(128,1))
        layer_temp.append(nn.Sigmoid())
        super().__init__(*layer_temp)
    

class Sembedencoder(nn.Sequential):
   

    def __init__(self,input_channels=1):
        layer_temp=[]
        layer_temp.append(nn.Conv1d(input_channels, out_channels=6,kernel_size=3,
                                        stride=1, padding=1))
        layer_temp.append(nn.ReLU())
        layer_temp.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))

        layer_temp.append(nn.Conv1d(6, out_channels=12,kernel_size=3,
                                        stride=1, padding=1))
        layer_temp.append(nn.ReLU())
        layer_temp.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))

        layer_temp.append(nn.Conv1d(12, out_channels=12,kernel_size=3,
                                        stride=1, padding=1))
        layer_temp.append(nn.ReLU())
        layer_temp.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))

        layer_temp.append(nn.Flatten(in_features=384,out_features=16))
        layer_temp.append(nn.Linear(in_features=384,out_features=16))
        # for i in range(2):
        #     if i >0:
        #         input_channels=inplanes
        #     layer_temp.append(nn.Conv1d(input_channels, inplanes,kernel_size=kernel_size,
        #                                 stride=stride, padding=(kernel_size-1)//2,bias=False))
        #     layer_temp.append(nn.BatchNorm1d(inplanes))
        #     layer_temp.append(nn.ReLU(inplace=True))
        #     layer_temp.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        # for i in range(2):
        #     layer_temp.append(nn.Conv1d(input_channels, inplanes,kernel_size=kernel_size,
        #                                 stride=stride, padding=(kernel_size-1)//2,bias=False))
        #     layer_temp.append(nn.BatchNorm1d(inplanes))
        #     layer_temp.append(nn.ReLU(inplace=True))
        super().__init__(*layer_temp)
    


    

# def RNet_1d(device,**kwargs):
#     model =RNet( **kwargs).to(device)

#     return model