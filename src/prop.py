from data_prop import *
from model import Property_prediction_deep
from util import *
import pytz
import torch
import os
import time
import copy
import random
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
from sklearn.metrics import mean_squared_error
from math import sqrt
from torch.utils.data.sampler import SubsetRandomSampler

UNDEFINED_INF = 1000000
USE_WEIGHTED_LOSS = False
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Normalizer(object):
    def __init__(self, tensor):
        self.mean = FloatTensor([torch.mean(tensor)])
        self.std = FloatTensor([torch.std(tensor)])

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

def sanitize(input_var, reference, return_diff=False):
    reference = torch.clamp(reference, max=UNDEFINED_INF)
    idx = (reference == UNDEFINED_INF).nonzero()
    idx = idx.view(-1)
    input_var_c = input_var.clone()
    if len(idx):
        reference.index_fill_(0, idx, 0)
        input_var_c.index_fill_(0, idx, 0)
    if return_diff:
        return input_var_c - reference
    else:
        return input_var_c, reference

def mae(prediction, target):
    return FloatTensor([torch.sum(torch.abs(sanitize(prediction, target, return_diff=True)))])

def sparse_loss(prop_model,input):
    parameters = list(prop_model.parameters())
    l1_regularization = torch.mean(torch.abs(parameters[0]))
    return l1_regularization

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='../data/', help='Data Path')
    parser.add_argument('--test-ratio', type=float, default=0.8, help='Test Split')
    parser.add_argument('--lrate', type=float, default=0.003, help='Learning Rate')
    parser.add_argument('--atom-feat', type=int, default=64, help='Atom Feature Dimension')
    parser.add_argument('--nconv', type=int, default=3, help='Number of Convolution Layers')
    parser.add_argument('--epoch', type=int, default=500, help='Number of Training Epoch')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--workers', type=int, default=0, help='workers')
    parser.add_argument('--radius', type=int, default=8, help='Radius of the sphere along an atom')
    parser.add_argument('--max-nbr', type=int, default=12, help='Maximum Number of neighbours to consider')
    parser.add_argument('--pretrained-model', type=str, default=None, help='Path of the Pretrained CrysAE Model Path')
    args = parser.parse_args()
    return args

def main(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    args = args_parse()
    eastern = pytz.timezone('Asia/Kolkata')
    current_time = datetime.datetime.now().astimezone(eastern).time()
    current_date = datetime.datetime.now().astimezone(eastern).date()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    atom_fea_len = args.atom_feat
    h_fea_len=args.atom_feat
    n_conv = args.nconv
    epochs = args.epoch
    lr = args.lrate
    lamda=0.01
    batch_size = args.batch_size
    weight_decay = 0.0
    workers = args.workers
    radius=args.radius
    max_num_nbr = args.max_nbr
    data_path = args.data_path
    pretrained_path=args.pretrained_model
    cuda = torch.cuda.is_available()
    print("Cuda enabled: ", cuda)
    test_ratio=args.test_ratio
    train_ratio = 1-test_ratio
    full_dataset = CIFData(data_path,max_num_nbr,radius)
    datasize=len(full_dataset)
    print("Data path :" + str(data_path))
    print("Radius :" + str(radius))
    print("Neighbours :" + str(max_num_nbr))
    print("Test Ratio :" + str(test_ratio))
    print("Train size " + str(datasize * train_ratio))
    print("Test size " + str(datasize * test_ratio))
    print("Pretrained Model Path " +str(pretrained_path))

    path = '../results/Prediction/'+ str(current_date) + '/' +str(test_ratio)+"_"+str(current_time)
    if not os.path.exists(path):
        os.makedirs(path)
    out = open(path + "/out.txt", "w")

    out.writelines("Data path : " + str(data_path))
    out.writelines("\n")
    out.writelines("Learning Rate : " + str(lr))
    out.writelines("\n")
    out.writelines("Radius :" + str(radius))
    out.writelines("\n")
    out.writelines("Neighbours :" + str(max_num_nbr))
    out.writelines("\n")
    out.writelines("Epochs " + str(epochs))
    out.writelines("\n")
    out.writelines("Lamda " + str(lamda))
    out.writelines("\n")
    out.writelines("Train size " + str(datasize * train_ratio))
    out.writelines("\n")
    out.writelines("Test size " + str(datasize * test_ratio))
    out.writelines("\n")
    out.writelines("\n")


    if len(full_dataset) < 2000:
        sample_data_list = [full_dataset[i] for i in tqdm(range(len(full_dataset)))]
    else:
        sample_data_list = [full_dataset[i] for i in tqdm(random.sample(range(len(full_dataset)), 2000))]

    _, sample_target= collate_pool(sample_data_list)
    normalizer = Normalizer(sample_target)
    print("Normalizer Property Mean  :" + str(normalizer.mean))
    print("Normalizer Property Stdev :" + str(normalizer.std))

    collate_fn = collate_pool
    idx_train, idx_test = train_test_split(range(datasize), test_size=test_ratio, random_state=42)
    train_data_loader = get_data_loader(dataset=full_dataset,collate_fn=collate_fn,train_size=idx_train,batch_size=batch_size,num_workers=workers,pin_memory=cuda)
    test_data_loader = get_data_loader(dataset=full_dataset,collate_fn=collate_fn,train_size=idx_test,batch_size=batch_size,num_workers=workers,pin_memory=cuda)

    print("Load Batch Data(Train/Test) Started .....")
    batch_data_train = []
    batch_target_train = []
    for i, (input, targets) in tqdm(enumerate(train_data_loader)):
        batch_data_train.append(input)
        batch_target_train.append(targets)

    batch_data_test = []
    batch_target_test = []
    for i, (input, targets) in tqdm(enumerate(test_data_loader)):
        batch_data_test.append(input)
        batch_target_test.append(targets)
    print("Load Batch Data(Train/Test) Ended .....")

    structures, _, _,_ = full_dataset[0]
    atom_fea, nbr_fea, nbr_fea_idx = structures[0], structures[1], structures[2]
    orig_atom_fea_len = atom_fea.shape[-1]
    nbr_fea_len = nbr_fea.shape[-1]
    prop_model = Property_prediction_deep(orig_atom_fea_len, nbr_fea_len, atom_fea_len=atom_fea_len,h_fea_len=h_fea_len, n_conv=n_conv)

    if pretrained_path!=None:
        model_name=pretrained_path

        if cuda :
            model = torch.load(model_name)
        else:
            model = torch.load(model_name, map_location=torch.device('cpu'))

        # Transfer Weights from Encoder of AutoEnc
        model_dict = model.state_dict()
        model_dict.pop('embedding.weight')
        model_dict.pop('fc_adj.weight')
        model_dict.pop('fc_adj.bias')
        model_dict.pop('fc_edge.weight')
        model_dict.pop('fc_edge.bias')
        model_dict.pop('fc_atom_feature.weight')
        model_dict.pop('fc_atom_feature.bias')
        model_dict.pop('fc1.weight')
        model_dict.pop('fc1.bias')
        pmodel_dict = prop_model.state_dict()
        pmodel_dict.update(model_dict)
        prop_model.load_state_dict(pmodel_dict)
    else:
        print("No Pretrained Model.Property Predictor will be trained from Scratch!!")

    # region Optimizer
    optimizer = optim.Adam(prop_model.parameters(), lr, weight_decay=weight_decay)

    loss_function = nn.MSELoss().to(device)
    best_test_mae = 999
    best_model=prop_model
    for epoch in tqdm(range(epochs)):
        prop_model.train()
        mae_list = []
        loss_list = []
        l1_loss_list = []

        # Training Model
        for i in range(len(batch_data_train)):
            loss = 0
            ae = 0
            input=batch_data_train[i]
            true_target=batch_target_train[i].to(device)
            targets_normed = normalizer.norm(true_target)
            targets_var = Variable(targets_normed).to(device)
            atom_fea, nbr_fea, nbr_fea_idx, crys_graph_idx = \
                input[0], input[1], input[2], input[3]

            input_var = (Variable(atom_fea.to(device)), Variable(nbr_fea.to(device)), nbr_fea_idx.to(device),
                         [crys_idx.to(device) for crys_idx in crys_graph_idx])
            target_predicted,_ = prop_model(*input_var)
            for j in range(len(target_predicted)):
                target_p=target_predicted[j]
                target_o=true_target[j]
                loss = loss+loss_function(targets_var[j], target_p)
                ae = ae + mae(normalizer.denorm(target_p.data), target_o)
            loss=loss/len(target_predicted)

            l1_regularization=sparse_loss(prop_model,input_var)
            loss = loss + lamda * l1_regularization

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ae=ae/len(target_predicted)
            mae_list.append(ae.item())
            loss_list.append(loss.item())
            l1_loss_list.append(l1_regularization.item())

        results='Epoch :'+str(epoch+1)\
                +' Train Loss : '+str(np.mean(loss_list))\
                +' Train MAE : '+str(np.mean(mae_list))

        # Testing
        mae_test_list = []
        for i in range(len(batch_data_test)):
            ae=0
            prop_model.eval()
            input = batch_data_test[i]
            true_target = batch_target_test[i].to(device)
            atom_fea, nbr_fea, nbr_fea_idx, crys_graph_idx = \
                input[0], input[1], input[2], input[3]

            input_var = (Variable(atom_fea.to(device)), Variable(nbr_fea.to(device)), nbr_fea_idx.to(device),
                         [crys_idx.to(device) for crys_idx in crys_graph_idx])
            target_predicted,_ = prop_model(*input_var)
            for j in range(len(target_predicted)):
                target_p = target_predicted[j]
                target_o = true_target[j]
                ae = ae + mae(normalizer.denorm(target_p.data), target_o)
            ae = ae / len(target_predicted)
            mae_test_list.append(ae.item())
        results = results + ' Test MAE : ' + str(np.mean(mae_test_list))
        mean_test_mae=np.mean(mae_test_list)
        if mean_test_mae<best_test_mae:
            best_test_mae=mean_test_mae
            best_test_epoch=epoch+1
            best_model=prop_model
        results = results + ' Best Test MAE : ' + str(best_test_mae)
        print(results)
        out.writelines(results)
        out.writelines("\n")
    out.writelines("\n")
    print("**************** Summary of the experiments ***********************")
    print("Data path :" + str(data_path))
    print("Test Ratio :" + str(test_ratio))
    print("Best Test MAE :" + str(best_test_mae) + " Best Test Epcoh :" + str(best_test_epoch))
    out.writelines("Best Test MAE :" + str(best_test_mae) + " Best Test Epcoh :" + str(best_test_epoch))
    torch.save(best_model,"../model/model_pp.pth")




if __name__=="__main__":
    seed=123
    torch.manual_seed(seed)
    np.random.seed(seed)
    main(seed)
