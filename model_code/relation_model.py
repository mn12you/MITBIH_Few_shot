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
    

class SembedNet(nn.Sequential):
   

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

        layer_temp.append(nn.Flatten())
        layer_temp.append(nn.Linear(in_features=384,out_features=16))
        layer_temp.append(nn.Linear(in_features=16,out_features=5))
        layer_temp.append(nn.Softmax(dim=1))
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

class transform_fully(nn.Sequential):
   

    def __init__(self,input_channels=1):
        layer_temp=[]
        layer_temp.append(nn.Linear(in_features=1,out_features=1))
        super().__init__(*layer_temp)

class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        layer_temp=[]
        layer_temp.append(nn.Conv1d(1, out_channels=6,kernel_size=3,
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

        layer_temp.append(nn.Flatten())
        layer_temp.append(nn.Linear(in_features=384,out_features=16))

        self.encoder = torch.nn.Sequential(*layer_temp)

        # add linear layers to compare between the features of the two images
        self.fc =nn.Linear(1, 1)

        self.sigmoid = nn.Sigmoid()

        
    # def init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         torch.nn.init.xavier_uniform_(m.weight)
    #         m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.encoder(x)
        output = output.view(output.size()[0], -1)

        return output

    def forward(self, input1, input2):

        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate both images' features
        # output = torch.cat((output1, output2), 1)
        output=torch.norm((output1-output2), p=1, dim=1).unsqueeze(-1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)

        
        return output
    

# def RNet_1d(device,**kwargs):
#     model =RNet( **kwargs).to(device)

#     return model



class SiameseNetwork_LMU(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        layer_temp=[]
        layer_temp.append(nn.Conv1d(1, out_channels=6,kernel_size=3,
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

        layer_temp.append(nn.Flatten())
        layer_temp.append(nn.Linear(in_features=384,out_features=16))

        self.encoder = torch.nn.Sequential(*layer_temp)

        # add linear layers to compare between the features of the two images
        self.fc =nn.Linear(1, 1)

        self.sigmoid = nn.Sigmoid()

        
    # def init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         torch.nn.init.xavier_uniform_(m.weight)
    #         m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.encoder(x)
        output = output.view(output.size()[0], -1)

        return output

    def forward(self, input1, input2):

        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate both images' features
        # output = torch.cat((output1, output2), 1)
        output=torch.norm((output1-output2), p=1, dim=1).unsqueeze(-1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)

        
        return output
    
