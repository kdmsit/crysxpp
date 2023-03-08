import argparse
import os
import sys
import pytz
import time
import torch
import shutil
import datetime
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from random import sample
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from data_prop import CIFData
from data_prop import collate_pool, get_train_val_test_loader
from model import CrystalGraphConvNet


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

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


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='../model/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '../model/model_best.pth.tar')


def args_parse():
    parser = argparse.ArgumentParser(description='Crystal Explainable Property Predictor')
    parser.add_argument('--data-path', type=str, default='../data/', help='Data Path')
    parser.add_argument('--radius', type=int, default=8, help='Radius of the sphere along an atom')
    parser.add_argument('--max-nbr', type=int, default=12, help='Maximum Number of neighbours to consider')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',help='number of data loading workers (default: 0)')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',help='number of total epochs to run (default: 30)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,metavar='LR', help='initial learning rate (default: 0.01)')
    parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,metavar='N', help='milestones for scheduler (default: [100])')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,metavar='N', help='print frequency (default: 10)')

    train_group = parser.add_mutually_exclusive_group()
    train_group.add_argument('--train-ratio', default=0.8, type=float, metavar='N',help='number of training data to be loaded (default none)')
    train_group.add_argument('--train-size', default=None, type=int, metavar='N',help='number of training data to be loaded (default none)')

    valid_group = parser.add_mutually_exclusive_group()
    valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N',help='percentage of validation data to be loaded (default 0.1)')
    valid_group.add_argument('--val-size', default=None, type=int, metavar='N',help='number of validation data to be loaded (default 1000)')

    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument('--test-ratio', default=0.2, type=float, metavar='N',help='percentage of test data to be loaded (default 0.1)')
    test_group.add_argument('--test-size', default=None, type=int, metavar='N',help='number of test data to be loaded (default 1000)')

    parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',help='choose an optimizer, SGD or Adam, (default: SGD)')
    parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',help='number of hidden atom features in conv layers')
    parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',help='number of hidden features after pooling')
    parser.add_argument('--n-conv', default=3, type=int, metavar='N',help='number of conv layers')
    parser.add_argument('--n-h', default=1, type=int, metavar='N',help='number of hidden layers after pooling')

    ## Added by Authors of CrysXPP
    parser.add_argument('--pretrained-model', type=str, default=None, help='Path of the Pretrained CrysAE Model Path')
    parser.add_argument('--feature-selector', type=bool, default=True, help='Option for feature selector')

    args = parser.parse_args(sys.argv[1:])
    return args



# # Sparse Loss Introduced by CrysXPP (Added by Authors of CrysXPP)
def sparse_loss(prop_model,input):
    parameters = list(prop_model.parameters())
    l1_regularization = torch.mean(torch.abs(parameters[0]))
    return l1_regularization

def main():
    args = args_parse()
    args.cuda = torch.cuda.is_available()
    best_mae_error = 1e10

    eastern = pytz.timezone('Asia/Kolkata')
    current_time = datetime.datetime.now().astimezone(eastern).time()
    current_date = datetime.datetime.now().astimezone(eastern).date()

    # load data
    dataset = CIFData(args.data_path,args.max_nbr,args.radius)
    collate_fn = collate_pool
    train_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        pin_memory=args.cuda,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size)

    # obtain target value normalizer
    if len(dataset) < 2000:
        sample_data_list = [dataset[i] for i in tqdm(range(len(dataset)))]
    else:
        sample_data_list = [dataset[i] for i in tqdm(sample(range(len(dataset)), 2000))]
    _, sample_target, _ = collate_pool(sample_data_list)
    normalizer = Normalizer(sample_target)

    # build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,atom_fea_len=args.atom_fea_len,n_conv=args.n_conv,
                                h_fea_len=args.h_fea_len,n_h=args.n_h,classification=False)
    if args.cuda:
        model.cuda()

    path = '../results/Prediction/' + str(current_date) + '/' + str(current_time)
    if not os.path.exists(path):
        os.makedirs(path)
    out = open(path + "/out.txt", "w")

    print("Pretrained Model Path " + str(args.pretrained_model))
    out.writelines("Pretrained Model Path " + str(args.pretrained_model))
    out.writelines("\n")
    print("Data path :" + str(args.data_path))
    out.writelines("Data path : " + str(args.data_path))
    out.writelines("\n")
    out.writelines("Learning Rate : " + str(args.lr))
    out.writelines("\n")
    print("Epochs " + str(args.epochs))
    out.writelines("Epochs " + str(args.epochs))
    out.writelines("\n")
    print("Train ratio " + str(args.train_ratio))
    out.writelines("Train ratio " + str(args.train_ratio))
    out.writelines("\n")
    print("Val ratio " + str(args.val_ratio))
    out.writelines("Val ratio " + str(args.val_ratio))
    out.writelines("\n")
    print("Test ratio " + str(args.test_ratio))
    out.writelines("Test ratio " + str(args.test_ratio))
    out.writelines("\n")
    print("Batch size " + str(args.batch_size))
    out.writelines("Batch size " + str(args.batch_size))
    out.writelines("\n")


    # Parameter Initialisation from Encoder of CrysAE (Added by Authors of CrysXPP)
    pretrained_path = args.pretrained_model
    if pretrained_path != None:
        model_name = pretrained_path
        if args.cuda:
            ae_model = torch.load(model_name)
        else:
            ae_model = torch.load(model_name, map_location=torch.device('cpu'))

        # Transfer Weights from Encoder of AutoEnc
        model_dict = ae_model.state_dict()
        model_dict.pop('embedding.weight')
        model_dict.pop('fc_adj.weight')
        model_dict.pop('fc_adj.bias')
        # model_dict.pop('fc_edge.weight')
        # model_dict.pop('fc_edge.bias')
        model_dict.pop('fc_atom_feature.weight')
        model_dict.pop('fc_atom_feature.bias')
        model_dict.pop('fc1.weight')
        model_dict.pop('fc1.bias')
        # model_dict.pop('fc2.weight')
        # model_dict.pop('fc2.bias')
        pmodel_dict = model.state_dict()
        pmodel_dict.update(model_dict)
        model.load_state_dict(pmodel_dict)
    else:
        print("No Pretrained Model.Property Predictor will be trained from Scratch!!")

    # define loss func and optimizer
    criterion = nn.MSELoss()

    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')

    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,gamma=0.1)

    for epoch in tqdm(range(args.epochs)):
        # train for one epoch
        train_loss, train_mae=train(train_loader, model, criterion, optimizer, normalizer,args,epoch,out)

        # evaluate on test set
        test_mae = validate(test_loader, model, criterion, normalizer,args,epoch,out)

        scheduler.step()

        # remember the best mae_eror and save checkpoint
        is_best = test_mae < best_mae_error
        if is_best:
            print("Saving Best Model on Validation....")
            best_mae_error = min(test_mae, best_mae_error)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'args': vars(args)
        }, is_best)
        results = 'Epoch Summary :'\
                  + ' Epoch :' + str(epoch + 1) \
                  + ' Train Mean Loss : ' + str(train_loss) \
                  + ' Train MAE : ' + str(train_mae) \
                  + ' Test MAE : ' + str(test_mae) \
                  + ' Best Test MAE : ' + str(best_mae_error)
        print("\n")
        print(results)
        out.writelines(results)
        out.writelines("\n")
        print("\n")

    print("Best Test MAE :" + str(best_mae_error))
    out.writelines("Best Test MAE :" + str(best_mae_error))

    # # test best model
    # print('---------Evaluate Model on Test Set---------------')
    # best_checkpoint = torch.load('../model/model_best.pth.tar')
    # model.load_state_dict(best_checkpoint['state_dict'])
    # test_mae=validate(test_loader, model, criterion, normalizer,args,epoch,out)
    # print("Test MAE on Best Validation Model :" + str(test_mae))
    # out.writelines("Test MAE on Best Validation Model :" + str(test_mae))


def train(train_loader, model, criterion, optimizer, normalizer,args,epoch,out):
    data_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3])

        # normalize target
        target_normed = normalizer.norm(target)

        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        # compute output
        output,_ = model(*input_var)
        # output = model(*input_var)
        loss = criterion(output, target_var)

        # L1 Regularised Loss for Sparse Feature Selection (Added by Authors of CrysXPP)
        # if args.feature_selector:
        #     l1_regularization = sparse_loss(model, input_var)
        #     loss = loss + 0.01 * l1_regularization


        # measure accuracy and record loss
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % args.print_freq == 0:
            results = 'Training Results : Epoch :' + str(epoch + 1) \
                      + ' Batch :' + str(i) \
                      + ' Train Loss : ' + str(losses.avg) \
                      + ' Train MAE : ' + str(mae_errors.avg)
            print(results)
            out.writelines(results)
            out.writelines("\n")

    return losses.avg, mae_errors.avg


def validate(val_loader, model, criterion, normalizer,args,epoch,out):
    losses = AverageMeter()
    mae_errors = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        if args.cuda:
            with torch.no_grad():
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            with torch.no_grad():
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])

        target_normed = normalizer.norm(target)

        if args.cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)

        # compute output
        output,_ = model(*input_var)
        # output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

        if i % args.print_freq == 0:
            results = 'Validation Results : Epoch :' + str(epoch + 1) \
                      + ' Batch :' + str(i) \
                      + ' Val Loss : ' + str(losses.avg) \
                      + ' Val MAE : ' + str(mae_errors.avg)
            print(results)
            out.writelines(results)
            out.writelines("\n")

    return mae_errors.avg





if __name__ == '__main__':
    main()
