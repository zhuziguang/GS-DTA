import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import time
import torch.nn as nn
from models.gat import GAT_GCN_Net
from models.gat_gcn import GAT_GCN_CNN_BiLSTM_Transformer
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from models.gatv2_gcn import GATv2_GCN_CNN_BiLSTM_Transformer
from utils import *

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


datasets = [['davis','kiba'][int(sys.argv[1])]] 
modeling = [GINConvNet, GAT_GCN_Net, GATv2_GCN_CNN_BiLSTM_Transformer, GCNNet][int(sys.argv[2])]
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = "cuda:" + str(int(sys.argv[3])) 
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE =512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 2500

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset )
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root='data', dataset=dataset+'_train')
        test_data = TestbedDataset(root='data', dataset=dataset+'_test')
        
        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_mse = 1000
        best_ci = 0
        best_epoch = -1
        model_file_name = 'model_' + model_st +'_' + dataset +  '.model'
        result_file_name = 'result_' + model_st +'_' + dataset +  '.csv'
        file_MSE = 'result' + model_st +'_' + dataset +   '.txt'
        MSEs = ('Epoch\tRMSE\tMSE\tPearson\tSpearman\tCi\trm2')
        with open(file_MSE, 'w') as f:
            f.write(MSEs + '\n')
        
        pretrained_model_file="/home/zhuziguang/kiba-quanzhong/MSE_0.21343421936035156_CI_0.9038705198445256_rm2_0.7093355381639808_1721181463.0038657.model"
        if os.path.isfile(pretrained_model_file):
            model.load_state_dict(torch.load(pretrained_model_file))
            print('Pretrained model loaded successfully!')
        else:
            print('No Pretrained model found!')
        
        for epoch in range(NUM_EPOCHS):
            train(model, device, train_loader, optimizer, epoch+1)
            G,P = predicting(model, device, test_loader)
            ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P),get_rm2(G.reshape(G.shape[0],-1),P.reshape(P.shape[0],-1))]
            RMSE=rmse(G,P)
            MSE=mse(G,P)
            Pearson=pearson(G,P)
            Spearson=spearman(G,P)
            Ci=ci(G,P)
            rm2=get_rm2(G.reshape(G.shape[0],-1),P.reshape(P.shape[0],-1))
            
            if  ret[-1]>=0.70:
                best_epoch = epoch+1
                best_mse = ret[1]
                best_ci = ret[-2]
                best_rm2=ret[-1]
                print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci,best_rm2:', best_mse,best_ci,best_rm2,model_st,dataset)
                #save_MSE_CI(MSE_CI,file_MSE)
                torch.save(model.state_dict(), f'./davis_result/MSE_{best_mse}_CI_{best_ci}_rm2_{best_rm2}_{time.time()}.model')
            else:
                print(ret[1],'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci,model_st,dataset)    
            
