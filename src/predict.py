import os
import torch
import random
import csv
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_prop import *
from model import CrystalGraphConvNet

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

def plot(epoch_list, loss_prop_list,path):
    frontsize=20
    line_width=5
    fig_path=path+'/Train losses_'
    df=pd.DataFrame({'x': epoch_list,'y1': loss_prop_list})
    plt.plot( 'x', 'y1', data=df, color='green', linewidth=line_width,label="Train Loss")
    plt.xlabel("Number of epochs ", fontsize=frontsize,labelpad=10,fontweight='bold')
    plt.ylabel("Losses", fontsize=frontsize,fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='../data_1000/', help='Data Path')
    parser.add_argument('--test-ratio', type=float, default=0.8, help='Test Split')
    parser.add_argument('--lrate', type=float, default=0.01, help='Learning Rate')
    parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',help='number of hidden atom features in conv layers')
    parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',help='number of hidden features after pooling')
    parser.add_argument('--n-conv', type=int, default=3, help='Number of Convolution Layers')
    parser.add_argument('--n-h', default=1, type=int, metavar='N', help='number of hidden layers after pooling')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--workers', type=int, default=0, help='workers')
    parser.add_argument('--radius', type=int, default=8, help='Radius of the sphere along an atom')
    parser.add_argument('--max-nbr', type=int, default=12, help='Maximum Number of neighbours to consider')
    parser.add_argument('--property-model', type=str, default='../model/model_best.pth.tar', help='Path of the Pretrained CrysXPP Model Path')
    args = parser.parse_args()
    return args

def main(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    args = args_parse()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = args.lrate
    batch_size = args.batch_size
    weight_decay = 0.0
    workers = args.workers
    radius=args.radius
    max_num_nbr = args.max_nbr
    data_path = args.data_path

    cuda = torch.cuda.is_available()
    print("Cuda enabled: ", cuda)
    full_dataset = CIFData(data_path,max_num_nbr,radius)
    datasize=len(full_dataset)



    if len(full_dataset) < 2000:
        sample_data_list = [full_dataset[i] for i in tqdm(range(len(full_dataset)))]
    else:
        sample_data_list = [full_dataset[i] for i in tqdm(random.sample(range(len(full_dataset)), 2000))]

    _, sample_target,_= collate_pool(sample_data_list)
    normalizer = Normalizer(sample_target)
    print("Normalizer Property Mean  :" + str(normalizer.mean))
    print("Normalizer Property Stdev :" + str(normalizer.std))

    collate_fn = collate_pool
    test_data_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.workers, collate_fn=collate_fn,
                             pin_memory=cuda)


    print("Load Batch Data(Test) Started .....")

    batch_data_test = []
    batch_target_test = []
    batch_cif = []
    for i, (input, targets,batch_cif_ids) in tqdm(enumerate(test_data_loader)):
        batch_data_test.append(input)
        batch_target_test.append(targets)
        batch_cif.append(batch_cif_ids)
    print("Load Batch Data(Train/Test) Ended .....")

    structures, _, _= full_dataset[0]
    atom_fea, nbr_fea, nbr_fea_idx = structures[0], structures[1], structures[2]
    orig_atom_fea_len = atom_fea.shape[-1]
    nbr_fea_len = nbr_fea.shape[-1]

    prop_model_name=args.property_model
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len, atom_fea_len=args.atom_fea_len, n_conv=args.n_conv,
                                h_fea_len=args.h_fea_len, n_h=args.n_h, classification=False)

    if cuda:
        model.cuda()

    # region Optimizer
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=weight_decay)


    if os.path.isfile(args.property_model ):
        print("=> loading model '{}'".format(args.property_model ))
        checkpoint = torch.load(args.property_model ,map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        normalizer.load_state_dict(checkpoint['normalizer'])
        print("=> loaded model '{}' (epoch {}, validation {})"
              .format(args.property_model , checkpoint['epoch'],
                      checkpoint['best_mae_error']))
    else:
        print("=> no model found at '{}'".format(args.property_model ))






    # Testing
    mae_test_list = []
    cif_ids = []
    test_targets = []
    test_preds = []
    for i in range(len(batch_data_test)):
        ae = 0
        model.eval()
        input = batch_data_test[i]
        cif_ids +=batch_cif[i]
        for target in batch_target_test[i]:
            test_targets.append(target.item())
        true_target = batch_target_test[i].to(device)
        atom_fea, nbr_fea, nbr_fea_idx, crys_graph_idx = input[0], input[1], input[2], input[3]

        input_var = (Variable(atom_fea.to(device)), Variable(nbr_fea.to(device)), nbr_fea_idx.to(device),
                     [crys_idx.to(device) for crys_idx in crys_graph_idx])
        target_predicted, _ = model(*input_var)
        for predict in target_predicted:
            test_preds.append(predict.item())
        for j in range(len(target_predicted)):
            target_p = target_predicted[j]
            target_o = true_target[j]
            ae = ae + mae(normalizer.denorm(target_p.data), target_o)
        ae = ae / len(target_predicted)
        mae_test_list.append(ae.item())
    results = ' Test MAE : ' + str(np.mean(mae_test_list))
    print(results)
    with open('test_results.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(("CIF Id", "Target", "Prediction"))
        for cif_id, target, pred in zip(cif_ids, test_targets,test_preds):
            writer.writerow((cif_id, target, pred))


if __name__=="__main__":
    seed=123
    torch.manual_seed(seed)
    np.random.seed(seed)
    main(seed)
