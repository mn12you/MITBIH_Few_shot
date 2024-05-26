import torch 

tensor = torch.arange(1, 10).float().view(9, 1)
tensor=tensor.unsqueeze(-1).repeat(1,1,16).squeeze(1)
tensor2 = torch.arange(11, 20).float().view(9, 1)
tensor2=tensor2.unsqueeze(-1).repeat(1,1,16).squeeze(1)
print(torch.norm((tensor-tensor2), p=1, dim=1).shape)
# tensor=tensor.unsqueeze(1).repeat(1,25,1).view(225,-1)
# tensor2 = torch.arange(11, 20).float().view(9, 1)
# tensor2=tensor2.unsqueeze(1).repeat(1,25,1).view(225,-1)
# print(tensor-tensor2)
#  distance=torch.norm((embed_query_ext-embed_support), p=1, dim=1)
# support=tensor.view(45*5*1,-1)
# aa=support.view(9,5,5,-1)
# output=aa.sum(dim=1).squeeze(-1)
# print(output)

# support,query,labels= support.float().to(device), query.float().to(device), labels.float().to(device)
# batch=labels.shape[0]
# class_num=labels.shape[-1]
# support=support.view(batch*class_num*shot,-1).unsqueeze(1)
# embed_query=encoder_net(query)
# embed_support=encoder_net(support)
# embed_query_ext=embed_query.unsqueeze(1).repeat(1,class_num*shot,1)
# embed_query_ext=embed_query_ext.view(batch*class_num*shot,-1)
# distance=torch.norm((embed_query_ext-embed_support), p=1, dim=1)
# distance=torch.unsqueeze(distance, 1)
# m=nn.Sigmoid()
# if shot>1:
#     output=m(net(distance)).view(batch,shot,class_num,-1)
#     output=output.sum(dim=1).squeeze(-1)
# else:
#     output=m(net(distance)).view(-1,class_num)
# output_list.append(output.data.cpu().numpy())
# labels_list.append(labels.data.cpu().numpy())