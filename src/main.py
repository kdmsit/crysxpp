from data import *
from model import CrystalAE
import pytz
import torch
import os
import time
import copy
import pickle as pkl
import datetime
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import networkx as nx
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

def plot(train_loss,lbl,yaxis_lbl,xaxis_lbl,path):
    frontsize = 30
    epoclist = [i for i in range(len(train_loss))]
    df = pd.DataFrame({'x': epoclist, 'y1': train_loss})
    plt.figure(figsize=(16, 14))
    plt.tight_layout()
    plt.plot('x', 'y1', data=df, marker='', color='blue', linewidth=10, label=lbl)
    plt.xlabel(xaxis_lbl, fontsize=frontsize)
    plt.ylabel(yaxis_lbl, fontsize=frontsize)
    plt.xticks(fontsize=frontsize)
    plt.yticks(fontsize=frontsize)
    plt.legend(fontsize=frontsize)
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.savefig(path)

def args_parse():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data-path', type=str, default='../../mp_2018/', help='Root Data Path')
    parser.add_argument('--data-path', type=str, default='../data_small/', help='Root Data Path')
    parser.add_argument('--lrate', type=float, default=0.003, help='Learning Rate')
    parser.add_argument('--atom-feat', type=int, default=64, help='Atom Feature Dimension')
    parser.add_argument('--nconv', type=int, default=3, help='Number of Convolution Layers')
    parser.add_argument('--epoch', type=int, default=200, help='Number of Training Epoch')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--workers', type=int, default=0, help='workers')
    parser.add_argument('--radius', type=int, default=8, help='Radius of the sphere along an atom')
    parser.add_argument('--max-nbr', type=int, default=12, help='Maximum Number of neighbours to consider')
    parser.add_argument('--is-global-loss', type=int, default=1, help='Flag for Global Connectivity Reconstruction Loss')
    parser.add_argument('--is-local-loss', type=int, default=1, help='Flag for Local Node Feature Reconstruction Loss')
    args = parser.parse_args()
    return args

def main(seed):

    #TODO : Put this part into argsparse
    eastern = pytz.timezone('Asia/Kolkata')
    current_time = datetime.datetime.now().astimezone(eastern).time()
    current_date= datetime.datetime.now().astimezone(eastern).date()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = args_parse()
    data_path = args.data_path
    atom_fea_len =args.atom_feat
    n_conv=args.nconv
    epochs =args.epoch
    lr =args.lrate
    batch_size=args.batch_size
    weight_decay=0.0
    workers=args.workers
    radius=args.radius
    max_num_nbr = args.max_nbr
    local_loss_flag=args.is_local_loss
    global_loss_flag = args.is_global_loss
    # print(local_loss_flag)
    # print(global_loss_flag)
    if local_loss_flag==0 and  global_loss_flag==0:
        print("Both Global and Local Loss flag is False, cant pretrain the autoencoder. Turn atleast one of them as True")
        exit()
    cuda = torch.cuda.is_available()
    print("Cuda enabled: ", cuda)
    print("Radius :" + str(radius))
    print("Neighbours :" + str(max_num_nbr))
    print("Start Time ", datetime.datetime.now())

    #Reading Full Datset
    full_dataset = CIFData(data_path,max_num_nbr,radius)
    datasize = len(full_dataset)
    train_size = datasize
    print(train_size)
    train_size = list(range(train_size))

    # Save in batch to avoid computation at each epoch
    collate_fn = collate_pool
    unsup_data_loader= get_data_loader(dataset=full_dataset,
                                            collate_fn=collate_fn,
                                            train_size=train_size,
                                            batch_size=batch_size,
                                            num_workers=workers,
                                            pin_memory=cuda)
    print("Sampled Data Size :",len(unsup_data_loader.batch_sampler.sampler.indices))

    print("Load Batch Data Started .....")
    batch_data_unsup=[]
    for i, (input, targets) in tqdm(enumerate(unsup_data_loader)):
        batch_data_unsup.append(input)
    print("Load Batch Train Data Ended .....")

    
    path='../results/CrystalAE/'+str(current_date)+'/'+str(current_time)
    if not os.path.exists(path):
        os.makedirs(path)


    out = open(path + "/out.txt","w")
    out.writelines("Adjacency List Vector Dimension : 4")
    out.writelines("\n")
    out.writelines("\n")
    out.writelines("***** Hyper-Parameters Details ********")
    out.writelines("\n")
    out.writelines("Radius :" + str(radius))
    out.writelines("\n")
    out.writelines("Neighbours :" + str(max_num_nbr))
    out.writelines("\n")
    out.writelines("atom_fea_len :" + str(atom_fea_len))
    out.writelines("\n")
    out.writelines("n_conv :" + str(n_conv))
    out.writelines("\n")
    out.writelines("epochs :" + str(epochs))
    out.writelines("\n")
    out.writelines("lr :" + str(lr))
    out.writelines("\n")
    out.writelines("batch_size :" + str(batch_size))
    out.writelines("\n")
    out.writelines("datasize :" + str(datasize))
    out.writelines("\n")


    # region Build model
    structures, _, _ ,_= full_dataset[0]
    atom_fea, nbr_fea, nbr_fea_idx=structures[0],structures[1],structures[2]
    orig_atom_fea_len = atom_fea.shape[-1]
    nbr_fea_len = nbr_fea.shape[-1]

    model = CrystalAE(orig_atom_fea_len, nbr_fea_len,atom_fea_len=atom_fea_len,n_conv=n_conv)
    save_model=model

    # region Optimizer
    optimizer = optim.Adam(model.parameters(), lr,weight_decay=weight_decay)
    # endregion

    # region Training Process (Stochastic)
    train_loss = []
    train_adj_reconst_loss = []
    train_edge_ftr_reconst_loss = []
    train_feature_reconst_loss = []

    mean_acc=[]
    min_loss=100
    for epoch in tqdm(range(epochs)):
        bt_loss_list = []
        bt_adj_reconst_loss_list = []
        bt_edge_ftr_reconst_loss_list = []
        bt_feature_reconst_loss_list = []
        train_acc = []
        print("\n")
        for input in batch_data_unsup:
            model.train()
            atom_fea, nbr_fea, nbr_fea_idx,adj_orig,edge_orig,batch_atom_fea,crys_graph_idx = \
                input[0], input[1], input[2],input[3],input[4],input[5],input[6]

            input_var = (Variable(atom_fea.to(device)), Variable(nbr_fea.to(device)), nbr_fea_idx.to(device),
                         [crys_idx.to(device) for crys_idx in crys_graph_idx])
            edge_prob_list, atom_feature_list, edge_distance_list = model(*input_var)
            loss=0
            loss_adj=0
            loss_edge=0
            loss_atom_feat=0
            for j in range(len(edge_prob_list)):
                if global_loss_flag:
                    # Loss for Global Connectivity Reconstruction
                    adj_p=edge_prob_list[j]
                    adj_o = adj_orig[j]
                    N = adj_p.size(0)
                    pos_weight = torch.Tensor([0.1, 1, 1, 1, 1, 1])
                    if torch.cuda.is_available():
                        pos_weight = pos_weight.cuda()
                        adj_o=adj_o.cuda()
                    loss_adj_reconst = F.nll_loss(adj_p, adj_o, weight=pos_weight).to(device)
                    loss_adj=loss_adj+loss_adj_reconst

                    # Loss for Edge Feature Reconstruction
                    edge_p = edge_distance_list[j]
                    edge_p = edge_p.view(N, 5)
                    edge_o = edge_orig[j].view(N, 5).to(device)
                    loss_edge_reconst = F.binary_cross_entropy_with_logits(edge_p, edge_o).to(device)
                    loss_edge = loss_edge + loss_edge_reconst
                    loss = loss + loss_adj_reconst+loss_edge_reconst

                if local_loss_flag:
                    #Loss for Local Atom Feature Reconstruction
                    atom_fea_p = atom_feature_list[j]
                    atom_feat_o = batch_atom_fea[j].to(device)
                    loss_atom_feat_reconst = F.binary_cross_entropy_with_logits(atom_fea_p, atom_feat_o).to(device)
                    loss_atom_feat = loss_atom_feat + loss_atom_feat_reconst
                    loss = loss  + loss_atom_feat_reconst

            loss=loss/batch_size


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bt_loss_list.append(loss.item())
            if global_loss_flag:
                loss_adj = loss_adj / batch_size
                loss_edge = loss_edge / batch_size
                bt_adj_reconst_loss_list.append(loss_adj.item())
                bt_edge_ftr_reconst_loss_list.append(loss_edge.item())
            if local_loss_flag:
                loss_atom_feat = loss_atom_feat / batch_size
                bt_feature_reconst_loss_list.append(loss_atom_feat.item())

        if local_loss_flag == 1 and global_loss_flag == 1:
            result = "Epoch " + str(epoch + 1) + \
                     ", Loss " + str(np.mean(bt_loss_list)) + \
                     ", Adjacency Reconstruction Loss " + str(np.mean(bt_adj_reconst_loss_list)) + \
                     ", Edge Feature Reconstruction Loss " + str(np.mean(bt_edge_ftr_reconst_loss_list)) + \
                     ", Feature Reconstruction " + str(np.mean(bt_feature_reconst_loss_list))
        elif local_loss_flag == 0:
            result = "Epoch " + str(epoch + 1) + \
                     ", Loss " + str(np.mean(bt_loss_list)) + \
                     ", Adjacency Reconstruction Loss " + str(np.mean(bt_adj_reconst_loss_list)) + \
                     ", Edge Feature Reconstruction Loss " + str(np.mean(bt_edge_ftr_reconst_loss_list))
        elif global_loss_flag == 0:
            result = "Epoch " + str(epoch + 1) + \
                     ", Loss " + str(np.mean(bt_loss_list)) + \
                     ", Feature Reconstruction " + str(np.mean(bt_feature_reconst_loss_list))

        train_loss.append(np.mean(bt_loss_list))
        train_adj_reconst_loss.append(np.mean(bt_adj_reconst_loss_list))
        train_edge_ftr_reconst_loss.append(np.mean(bt_edge_ftr_reconst_loss_list))
        train_feature_reconst_loss.append(np.mean(bt_feature_reconst_loss_list))
        mean_acc.append(np.mean(train_acc))
        print(result)
        out.writelines(result)
        out.writelines("\n")
        if np.mean(train_loss) < min_loss:
            min_loss = np.mean(train_loss)
            save_model = model
        if (epoch+1) % 30==0:
            torch.save(save_model,"../model/model_model_pretrain_" + str(epoch+1) + ".pth")
    plot(train_loss, 'Train Loss Plot', 'Reconstruction Total Loss', 'Number of epoch', path+'/train_loss.png')
    if global_loss_flag:
        plot(train_adj_reconst_loss, 'Adjacecny Reconstruction Loss Plot', 'Adjacecny Reconstruction Loss', 'Number of epoch',
             path + '/adj_reconstr_loss.png')
        plot(train_edge_ftr_reconst_loss, 'Edge Feature Reconstruction Loss Plot', 'Edge Feature Reconstruction Loss',
             'Number of epoch',
             path + '/edge_ftr_reconstr_loss.png')
    if local_loss_flag:
        plot(train_feature_reconst_loss, 'Node Feature Reconstruction Loss Plot', 'Node Feature Reconstruction Loss', 'Number of epoch',
             path + '/feature_reconstr_loss.png')
    torch.save(save_model, "../model/model_pretrain.pth")




    # endregion


if __name__=="__main__":
    seed=123
    torch.manual_seed(seed)
    np.random.seed(seed)
    main(seed)
