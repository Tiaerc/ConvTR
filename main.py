from torch.optim import optimizer
from torch.utils.data import DataLoader,Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from dataset import HSIDataset
from model import *
import urllib.request
import zipfile
import numpy as np
import torch
from collections import defaultdict
import argparse
import os
from torch.utils.tensorboard import SummaryWriter   

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def mae(y_true, y_pred):
    return torch.mean(torch.abs(y_pred - y_true))

def mape(y_true, y_pred, threshold=0.1):
    v = torch.clip(torch.abs(y_true), threshold, None)
    diff = torch.abs((y_true - y_pred) / v)
    return 100.0 * torch.mean(diff, axis=-1).mean()

def rse(y_true, y_pred):
    batch_num, j = y_true.shape
    return torch.sqrt(torch.square(y_pred - y_true).sum()/(batch_num-2))

def Print_loss(name, RMES, MAE, MAPE, RSE, m):
    print("%s_dataset sum loss: RMSE:%f, MAE:%f, MAPE:%f, RSE:%f" %(name, RMES, MAE/m, MAPE/m, RSE/m))


if __name__ == "__main__":
    seed_torch(222)
    writter = SummaryWriter("./events")
    parser = argparse.ArgumentParser()
    parser.add_argument("--Batch_size", type=int, default=256, nargs="?", help="Batch size.")
    parser.add_argument("--epochs", type=int, default=400, nargs="?", help="epochs")
    parser.add_argument("--nc", type=int, default=180, nargs="?", help="Channel")
    parser.add_argument("--TR_ranks", type=int, default=40, nargs="?", help="TR-ranks")
    parser.add_argument("--model", type=int, default=2, nargs="?", help="计算方法")
    parser.add_argument("--input_dropout", type=float, default=0.26694419227220374, nargs="?", help="Input layer dropout.")
    parser.add_argument("--hidden_dropout", type=float, default=0.2, nargs="?", help="Hidden layer dropout.")

    args = parser.parse_args()
    kwargs = {'input_dropout': args.input_dropout, 'hidden_dropout': args.hidden_dropout}
    torch.backends.cudnn.deterministic = True
    # seed = 1
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    # shape = torch.tensor((185, 290, 30))
    shape = torch.tensor((144, 176, 30))
    # shape = torch.tensor((256, 256, 30))
    HSI_data_train= HSIDataset('video30/suzie.mat', 0.12, 0.9, 0.1, 'train')
    HSI_data_test = HSIDataset('video30/suzie.mat', 0.12, 0.9, 0.1, 'test')

    # shape = torch.tensor((256, 256, 3))
    # HSI_data_train= HSIDataset('airplane.mat', 0.17, 0.9, 0.1, 'train')
    # HSI_data_test = HSIDataset('airplane.mat', 0.17, 0.9, 0.1, 'test')

    # shape = torch.tensor((256, 256, 11))
    # HSI_data_train= HSIDataset('WashtonDC.mat', 0.01, 0.9, 0.1, 'train')
    # HSI_data_test = HSIDataset('WashtonDC.mat', 0.01, 0.9, 0.1, 'test')
    
    # 构建DataLoader
    # HSI_loader = DataLoader(dataset=HSI_data, batch_size=args.Batch_size,shuffle=True)
    HSI_loader_train = DataLoader(dataset=HSI_data_train, batch_size=args.Batch_size,shuffle=True)
    HSI_loader_test = DataLoader(dataset=HSI_data_test, batch_size=args.Batch_size,shuffle=True)

    model = TRDnet(shape, args.TR_ranks, args.nc, args.model, **kwargs) 
    model = model.cuda() #第一句话

    # opt = torch.optim.SGD(model.parameters(), lr=1e-3,weight_decay=1e-2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss_train=list()
    loss_test=list()

    for i in range(0, args.epochs):

        rmse_tr, mae_tr, mape_tr, rse_tr= 0, 0, 0, 0 
        rmse_te, mae_te, mape_te, rse_te= 0, 0, 0, 0

        model.train()
        # 取训练集的数据进行修正
        for idex, label in HSI_loader_train:

            idex = idex.cuda() #第二句话
            label = label.cuda() #第三句话
            
            out = model(idex, args.TR_ranks) 
            # myloss = MyLoss()
            mseloss = torch.nn.MSELoss()
            out = torch.reshape(out, (-1, 1))

            label = torch.reshape(label, (-1, 1))
            label = label.to(torch.float32)
            # loss = myloss(out, label)
            loss = torch.sqrt(mseloss(out, label))

            opt.zero_grad()
            loss.backward()
            opt.step()

            rmse_tr += loss.item()
            mae_tr += mae(label, out).item()
            mape_tr += mape(label, out).item()
            rse_tr += rse(label, out).item()
            
        rmse_tr = rmse_tr/len(HSI_loader_train)
        writter.add_scalar("rmse_tr", rmse_tr, global_step=i)
        # print("sum loss:", sumloss / sumnum)
        Print_loss('Train',rmse_tr,mae_tr,mape_tr, rse_tr,len(HSI_loader_train))
        loss_train.append(round(rmse_tr,4 ))
            

        if (i % 5) == 0:
            # 保存参数
            # torch.save(model.state_dict(), "./out/model_2.pyt")
            # 保存整个模型
            # torch.save(model, "./Picture/Airplane/mr0.85/r20_nc180_dr0.2_.pyt")
            torch.save(model, "./video_result/mr0.90/suzie_r40_nc180_dr0.1_.pyt")
            

        with torch.no_grad():
            model.eval()
            for  idex_2, label_2 in HSI_loader_test:
                idex_2 = idex_2.cuda() 
                label_2 = label_2.cuda() 
                
                out_2 = model(idex_2, args.TR_ranks)
                mseloss = torch.nn.MSELoss()

                out_2 = torch.reshape(out_2, (-1, 1))
                label_2 = torch.reshape(label_2, (-1, 1))
                label_2 = label_2.to(torch.float32)
                loss_2 = torch.sqrt(mseloss(out_2, label_2))

                rmse_te += loss_2.item()
                mae_te += mae(label_2, out_2).item()
                mape_te += mape(label_2, out_2).item()
                rse_te += rse(label_2, out_2).item()

            rmse_te =rmse_te/len(HSI_loader_test)
            Print_loss('Test',rmse_te, mae_te, mape_te, rse_te, len(HSI_loader_test))
            loss_test.append(round(rmse_te,4 ))
            writter.add_scalar("rmes_tr_test", rmse_te, global_step=i)
            
        print("epoch:", i+1, "end")   

    print("loss_train:", loss_train)
    print("loss_test:" , loss_test)

