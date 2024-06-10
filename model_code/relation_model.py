import torch
import torch.nn as nn
import os
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
        super().__init__(*layer_temp)


class LMUEBCNet(nn.Sequential):
    
    def __init__(self):
        layer_temp=[]
        layer_temp.append(nn.Conv1d(1, out_channels=6,kernel_size=3,
                                        stride=1, padding=1))
        layer_temp.append(nn.ReLU())
        layer_temp.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))

        layer_temp.append(nn.Conv1d(6, out_channels=8,kernel_size=3,
                                        stride=1, padding=1))
        layer_temp.append(nn.ReLU())
        layer_temp.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))

        layer_temp.append(nn.Conv1d(8, out_channels=12,kernel_size=3,
                                        stride=1, padding=1))
        layer_temp.append(nn.Conv1d(12, out_channels=12,kernel_size=3,
                                        stride=1, padding=1))
        layer_temp.append(nn.ReLU())
        # layer_temp.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))

        layer_temp.append(nn.Conv1d(12, out_channels=12,kernel_size=3,
                                        stride=1, padding=1))
        layer_temp.append(nn.Conv1d(12, out_channels=12,kernel_size=3,
                                        stride=1, padding=1))
        layer_temp.append(nn.ReLU())
        layer_temp.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))

        layer_temp.append(nn.Flatten())
        layer_temp.append(nn.Linear(in_features=384,out_features=16))
        layer_temp.append(nn.Linear(in_features=16,out_features=5))
        layer_temp.append(nn.Softmax(dim=1))
        super().__init__(*layer_temp)



class Siamese_Sembed(nn.Module):

    def __init__(self):
        super(Siamese_Sembed, self).__init__()
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

    def forward_once(self, x):
        output = self.encoder(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output=torch.norm((output1-output2), p=1, dim=1).unsqueeze(-1)
        # pass the concatenation to the linear layers
        output = self.fc(output)
        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)
        return output
class Siamese_LMU(nn.Module):

    def __init__(self):
        super(Siamese_LMU, self).__init__()
        layer_temp=[]
        layer_temp.append(nn.Conv1d(1, out_channels=6,kernel_size=3,
                                        stride=1, padding=1))
        layer_temp.append(nn.ReLU())
        layer_temp.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))

        layer_temp.append(nn.Conv1d(6, out_channels=8,kernel_size=3,
                                        stride=1, padding=1))
        layer_temp.append(nn.ReLU())
        layer_temp.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))

        layer_temp.append(nn.Conv1d(8, out_channels=12,kernel_size=3,
                                        stride=1, padding=1))
        layer_temp.append(nn.Conv1d(12, out_channels=12,kernel_size=3,
                                        stride=1, padding=1))
        layer_temp.append(nn.ReLU())
        # layer_temp.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))

        layer_temp.append(nn.Conv1d(12, out_channels=12,kernel_size=3,
                                        stride=1, padding=1))
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

    def forward_once(self, x):
        output = self.encoder(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):

        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output=torch.norm((output1-output2), p=1, dim=1).unsqueeze(-1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)

        
        return output
    

class Siamese_Sembed_2D(nn.Module):

    def __init__(self):
        super(Siamese_Sembed_2D, self).__init__()
        layer_temp=[]
        layer_temp.append(nn.Conv2d(1, out_channels=6,kernel_size=3,
                                        stride=1, padding=1))
        layer_temp.append(nn.ReLU())
        layer_temp.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        layer_temp.append(nn.Conv2d(6, out_channels=12,kernel_size=3,
                                        stride=1, padding=1))
        layer_temp.append(nn.ReLU())
        layer_temp.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        layer_temp.append(nn.Conv2d(12, out_channels=12,kernel_size=3,
                                        stride=1, padding=1))
        layer_temp.append(nn.ReLU())
        layer_temp.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        layer_temp.append(nn.Flatten())
        layer_temp.append(nn.Linear(in_features=588,out_features=16))
        self.encoder = torch.nn.Sequential(*layer_temp)
        # add linear layers to compare between the features of the two images
        self.fc =nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward_once(self, x):
        output = self.encoder(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output=torch.norm((output1-output2), p=1, dim=1).unsqueeze(-1)
        # pass the concatenation to the linear layers
        output = self.fc(output)
        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)
        return output




class SembedNet_2D(nn.Sequential):
   

    def __init__(self,input_channels=1):
        layer_temp=[]
        layer_temp.append(nn.Conv2d(input_channels, out_channels=6,kernel_size=3,
                                      stride=1, padding=1))
        layer_temp.append(nn.ReLU())
        layer_temp.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        layer_temp.append(nn.Conv2d(6, out_channels=12,kernel_size=3,
                                        stride=1, padding=1))
        layer_temp.append(nn.ReLU())
        layer_temp.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        layer_temp.append(nn.Conv2d(12, out_channels=12,kernel_size=3,
                                        stride=1, padding=1))
        layer_temp.append(nn.ReLU())
        layer_temp.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        layer_temp.append(nn.Flatten())
        layer_temp.append(nn.Linear(in_features=588,out_features=16))
        layer_temp.append(nn.Linear(in_features=16,out_features=5))
        layer_temp.append(nn.Softmax(dim=1))
        super().__init__(*layer_temp)


class Siamese_CNN(nn.Module):

    def __init__(self):
        super(Siamese_CNN, self).__init__()
        layer_temp=[]
        layer_temp.append(nn.Conv1d(1, out_channels=64,kernel_size=10,
                                        stride=1, padding=0))
        layer_temp.append(nn.ReLU())
        layer_temp.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))

        layer_temp.append(nn.Conv1d(64, out_channels=128,kernel_size=7,
                                        stride=1, padding=0))
        layer_temp.append(nn.ReLU())
        layer_temp.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))

        layer_temp.append(nn.Conv1d(128, out_channels=128,kernel_size=4,
                                        stride=1, padding=0))
        layer_temp.append(nn.ReLU())
        layer_temp.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))

        layer_temp.append(nn.Conv1d(128, out_channels=256,kernel_size=4,
                                        stride=1, padding=0))
        layer_temp.append(nn.ReLU())
        layer_temp.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))

        layer_temp.append(nn.Flatten())
        layer_temp.append(nn.Linear(in_features=3072,out_features=2048))

        self.encoder = torch.nn.Sequential(*layer_temp)

        # add linear layers to compare between the features of the two images
        self.fc =nn.Linear(1, 1)

        self.sigmoid = nn.Sigmoid()

    def forward_once(self, x):
        output = self.encoder(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):

        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output=torch.norm((output1-output2), p=1, dim=1).unsqueeze(-1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)

        
        return output
    

class CNN(nn.Sequential):

    def __init__(self):
        layer_temp=[]
        layer_temp.append(nn.Conv1d(1, out_channels=64,kernel_size=10,
                                        stride=1, padding=0))
        layer_temp.append(nn.ReLU())
        layer_temp.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))

        layer_temp.append(nn.Conv1d(64, out_channels=128,kernel_size=7,
                                        stride=1, padding=0))
        layer_temp.append(nn.ReLU())
        layer_temp.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))

        layer_temp.append(nn.Conv1d(128, out_channels=128,kernel_size=4,
                                        stride=1, padding=0))
        layer_temp.append(nn.ReLU())
        layer_temp.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))

        layer_temp.append(nn.Conv1d(128, out_channels=256,kernel_size=4,
                                        stride=1, padding=0))
        layer_temp.append(nn.ReLU())
        layer_temp.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))

        layer_temp.append(nn.Flatten())
        layer_temp.append(nn.Linear(in_features=3072,out_features=2048))
        layer_temp.append(nn.Linear(in_features=2048,out_features=5))
        layer_temp.append(nn.Softmax(dim=1))
        super().__init__(*layer_temp)

    
    
