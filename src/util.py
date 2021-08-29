import math
import torch.nn.functional as F
from matplotlib import pyplot as plt


def plot_hist(data_list,filepath):
    data = data_list
    plt.ylim(0, 500)
    plt.bar(data.keys(), data.values(), 2, color='g')
    plt.title('Material data Atoms Count')
    plt.xlabel('Number of atoms per unit cell')
    plt.ylabel('count')
    plt.savefig(filepath)
    plt.close()

def accuracy(output, labels):
    # N=int(math.sqrt(output.size(0)))
    output = F.log_softmax(output,dim=1)
    preds = output.max(1)[1].type_as(labels)
    # labels = labels.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    accuracy = correct / len(labels)
    # labels = labels.view(N,N)
    # preds = preds.view(N,N)
    return accuracy,preds,labels