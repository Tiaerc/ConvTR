import numpy as np
from numpy import dtype, random, sqrt
from numpy.core.fromnumeric import trace
import torch
from torch.nn.init import xavier_normal_
import torch.nn.functional as F
# from torch.nn.modules.conv import Conv2d
from torch.utils.data import dataloader


class TRDnet(torch.nn.Module):
    def __init__(self, shape, ranks, nc, model, device="cuda", **kwargs):
        super(TRDnet, self).__init__()
        self.model = model

        self.N1 = torch.nn.Embedding(shape[0], embedding_dim=(ranks*ranks), padding_idx=0)
        self.N2 = torch.nn.Embedding(shape[1], embedding_dim=(ranks*ranks), padding_idx=0)
        self.N3 = torch.nn.Embedding(shape[2], embedding_dim=(ranks*ranks), padding_idx=0)
        self.N1.weight.data = (torch.randn((shape[0], (ranks*ranks)), dtype=torch.float).to(device))
        self.N2.weight.data = (torch.randn((shape[1], (ranks*ranks)), dtype=torch.float).to(device))
        self.N3.weight.data = (torch.randn((shape[2], (ranks*ranks)), dtype=torch.float).to(device))
        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout = torch.nn.Dropout(kwargs["hidden_dropout"])
        self.bne = torch.nn.BatchNorm2d(nc)

        #传统2层 先(R,R)  
        # self.conv2d_1 = torch.nn.Conv2d(1, nc, kernel_size=(ranks,ranks), stride = ranks)
        # self.conv2d_2 = torch.nn.Conv2d(nc, nc, kernel_size=(1, len(shape)))
        
        #传统2层 先(1,N)
        # self.conv2d_1 = torch.nn.Conv2d(1, nc, kernel_size=(1, len(shape)), stride = [1, len(shape)])
        # self.conv2d_2 = torch.nn.Conv2d(nc, nc, kernel_size=(ranks,ranks))

        # #先(1,N) 3层
        # self.conv2d_1 = torch.nn.Conv2d(1, nc, kernel_size=(1, len(shape)), stride = [1, len(shape)])
        # self.conv2d_2 = torch.nn.Conv2d(nc, nc, kernel_size=(ranks//2,ranks//2),stride = ranks//2)
        # self.conv2d_3 = torch.nn.Conv2d(nc, nc, kernel_size=(2,2))

        #先(1,N) 5层
        # self.conv2d_1 = torch.nn.Conv2d(1, nc, kernel_size=(1, len(shape)), stride = [1, len(shape)])
        # self.conv2d_2 = torch.nn.Conv2d(nc, nc, kernel_size=(ranks//4,ranks//4),stride = ranks//4)
        # self.conv2d_3 = torch.nn.Conv2d(nc, nc, kernel_size=(2,2))
        # self.conv2d_4 = torch.nn.Conv2d(nc, nc, kernel_size=(2,2))
        # self.conv2d_5 = torch.nn.Conv2d(nc, nc, kernel_size=(2,2))

        # #先(R,R) 4层  目前效果好
        # self.conv2d_1 = torch.nn.Conv2d(1, nc, kernel_size=(ranks//2,ranks//2), stride = ranks//2)
        # self.conv2d_2 = torch.nn.Conv2d(nc, nc, kernel_size=(1, len(shape)))
        # self.conv2d_3 = torch.nn.Conv2d(nc, nc, kernel_size=(2, 2),stride = 2)
        # self.conv2d_4 = torch.nn.Conv2d(nc, nc, kernel_size=(1, 2))

        #先(R,R) 4层 
        self.conv2d_1 = torch.nn.Conv2d(1, nc, kernel_size=(ranks//2,ranks//2), stride = ranks//2)
        self.conv2d_2 = torch.nn.Conv2d(nc, nc, kernel_size=(1, len(shape)))
        self.conv2d_3 = torch.nn.Conv2d(nc, nc, kernel_size=(2, 2))
        self.conv2d_4 = torch.nn.Conv2d(nc, nc, kernel_size=(1, len(shape)))

        # #先(R,R) 5/6层 
        # self.conv2d_1 = torch.nn.Conv2d(1, nc, kernel_size=(ranks//4,ranks//4), stride = ranks//4)
        # self.conv2d_2 = torch.nn.Conv2d(nc, nc, kernel_size=(1, len(shape)))
        # self.conv2d_3 = torch.nn.Conv2d(nc, nc, kernel_size=(2, 2))
        # self.conv2d_4 = torch.nn.Conv2d(nc, nc, kernel_size=(1, len(shape)), stride = [1, len(shape)])
        # # self.conv2d_5 = torch.nn.Conv2d(nc, nc, kernel_size=(2, 2))
        # # self.conv2d_6 = torch.nn.Conv2d(nc, nc, kernel_size=(2, 2))
        # self.conv2d_5 = torch.nn.Conv2d(nc, nc, kernel_size=(3, 3))

        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(in_features=nc, out_features=(nc//2))
        self.linear2 = torch.nn.Linear(in_features=(nc//2), out_features=1)



    def forward(self, idex, ranks):

        conv_Z = list()
        conv_Z_exist = False
        # tol_z = list() #用来储存batch_size个idex的切片
        for i in range(len(idex)):  #i=batch_size
            N1 = self.N1(idex[i][0])
            N2 = self.N2(idex[i][1])
            N3 = self.N3(idex[i][2])

            N1 = N1.view(ranks,ranks)
            N2 = N2.view(ranks,ranks)
            N3 = N3.view(ranks,ranks)
            k = torch.cat((N1, N2,N3), dim=1).unsqueeze(0).unsqueeze(0)
            # print(k,N1,N2,N3)
            if conv_Z_exist == False:
                    conv_Z = k
                    conv_Z_exist = True
            else:
                conv_Z = torch.cat((conv_Z,k),dim=0)

        
        if self.model == 1:
            pass
        
        if self.model == 2:
            #复原方法二：仿照CoSTCo
            # 1
            rst = self.conv2d_1(conv_Z)
            rst = self.bne(rst)
            rst = F.relu(rst)

            # 2
            rst = self.hidden_dropout(rst)
            rst = self.conv2d_2(rst) 
            # rst = self.bne(rst)
            rst = F.relu(rst)   

            # 3
            rst = self.hidden_dropout(rst)
            rst = self.conv2d_3(rst)  
            # rst = self.bne(rst)
            rst = F.relu(rst)   
            
            # 4
            # rst = self.hidden_dropout(rst)
            rst = self.conv2d_4(rst)  
            # rst = self.bne(rst)
            rst = F.relu(rst)   

            # # 5
            # # rst = self.hidden_dropout(rst)
            # rst = self.conv2d_5(rst)  
            # # rst = self.bne(rst)
            # rst = F.relu(rst)   

            # # 6
            # rst = self.hidden_dropout(rst)    
            # rst = self.conv2d_6(rst)  
            # # rst = self.bne(rst)
            # rst = F.relu(rst)  

            # rst = self.hidden_dropout(rst)    
            rst = self.flatten(rst) 
            rst = self.linear1(rst) 
            rst = F.relu(rst)

            rst = self.linear2(rst) 

            return rst

