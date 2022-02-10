from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def randomSeed(random_seed):
	"""Given a random seed, this will help reproduce results across runs"""
	if random_seed is not None:
		torch.manual_seed(random_seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(random_seed)


def getActivation(activation):
	if activation == 'softplus':
		return nn.Softplus()
	elif activation == 'relu':
		return nn.ReLU()


class ConvLayer(nn.Module):
	"""
	Convolutional operation on Crystal graphs (CGCNN Paper)
	"""
	def __init__(self, atom_fea_len, nbr_fea_len, random_seed=None, activation='relu'):
		randomSeed(random_seed)
		super(ConvLayer, self).__init__()
		self.atom_fea_len = atom_fea_len
		self.nbr_fea_len = nbr_fea_len
		self.fc_full = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len,2 * self.atom_fea_len).to(device)
		self.sigmoid = nn.Sigmoid().to(device)
		self.activation1 = getActivation(activation).to(device)
		self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len).to(device)
		self.bn2 = nn.BatchNorm1d(self.atom_fea_len).to(device)
		self.activation2 = getActivation(activation).to(device)

	def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
		N, M = nbr_fea_idx.shape
		# convolution
		atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]		# [N, M, atom_fea_len]
		atom_in_fea_exp=atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len)
		total_nbr_fea = torch.cat([atom_in_fea_exp,atom_nbr_fea, nbr_fea], dim=2)		# [N, M, nbr_fea_len + 2*atom_fea_len]

		total_gated_fea = self.fc_full(total_nbr_fea)		# [N, M, 2*atom_fea_len]

		total_gated_fea = self.bn1(total_gated_fea.view(-1, self.atom_fea_len * 2)).view(N, M, self.atom_fea_len * 2)

		nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)		# [N, M, atom_fea_len] each
		nbr_filter = self.sigmoid(nbr_filter)
		nbr_core = self.activation1(nbr_core)
		nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)		# [N, atom_fea_len]
		nbr_sumed = self.bn2(nbr_sumed)
		out = self.activation2(atom_in_fea + nbr_sumed)
		return out


class CrystalAE(nn.Module):
	"""
	Create a crystal graph auto encoder to learn node representations through unsupervised training
	"""

	def __init__(self, orig_atom_fea_len, nbr_fea_len,atom_fea_len=64, n_conv=3,activation='softplus',random_seed=None):
		randomSeed(random_seed)
		super(CrystalAE, self).__init__()
		self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len,bias=False).to(device)
		self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,nbr_fea_len=nbr_fea_len, random_seed=random_seed,activation=activation)
									for _ in range(n_conv)])
		self.fc_adj = nn.Bilinear(atom_fea_len,atom_fea_len, 6).to(device)
		self.fc1 = nn.Linear(6, 6).to(device)

		self.fc_edge = nn.Bilinear(atom_fea_len, atom_fea_len, 5).to(device)
		self.fc2 = nn.Linear(5, 5).to(device)


		self.fc_atom_feature = nn.Linear(atom_fea_len, orig_atom_fea_len).to(device)



	def forward(self, atom_fea, nbr_fea, nbr_fea_idx,crystal_atom_idx):
		# Encoder Part (Crystal Graph Convolution Encoder )
		atom_fea = self.embedding(atom_fea)
		for conv_func in self.convs:
			atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
		atom_emb = atom_fea.clone()

		bt_atom_fea = [atom_fea[idx_map] for idx_map in crystal_atom_idx]


		#Decoder Part
		'''
			Architecture : Node Emb is N*64. We decode it back to adjacency tensor N*N*4.
						   Entry (i,j) is a 4 dim one hot vector,
						   where 0th vector = 1 : No Edges
						   		 1st vector = 1 : 1/2 Edges (Low Connectivity)
						         2nd vector = 1 : 3/4 Edges (Medium Connectivity)
						         3rd vector = 1 : more than 5 Edges (High Connectivity)

		'''
		edge_prob_list=[]
		edge_feature_list=[]
		atom_feature_list=[]
		for atom_fea in bt_atom_fea:
			N=atom_fea.shape[0]
			dim = atom_fea.shape[1]

			# Repeat feature N times : (N,N,dim)
			atom_nbr_fea = atom_fea.repeat(N, 1, 1)
			atom_nbr_fea = atom_nbr_fea.contiguous().view(-1, dim)

			# Expand N times : (N,N,dim)
			atom_adj_fea = torch.unsqueeze(atom_fea, 1).expand(N, N, dim)
			atom_adj_fea = atom_adj_fea.contiguous().view(-1, dim)

			# Bilinear Layer : Adjacency List Reconstruction
			edge_p = self.fc_adj(atom_adj_fea, atom_nbr_fea)
			edge_p = self.fc1(edge_p)
			edge_p = F.log_softmax(edge_p, dim=1)
			edge_prob_list.append(edge_p)

			# Bilinear Layer : Edge Feature Reconstruction
			edge_fea = self.fc_edge(atom_adj_fea, atom_nbr_fea)
			edge_fea = self.fc2(edge_fea)
			edge_fea = edge_fea.view(N, N, 5)
			edge_feature_list.append(edge_fea)

			# Atom Feature Reconstruction
			atom_feature_list.append(self.fc_atom_feature(atom_fea))
		return edge_prob_list, atom_feature_list,edge_feature_list



class Property_prediction_deep(nn.Module):
	"""
		Property prediction model from the trained node embeddings, using Deep Layer
		"""
	def __init__(self,orig_atom_fea_len, nbr_fea_len,atom_fea_len=64,h_fea_len=64,n_conv=3,activation='softplus',random_seed=None):  #softplus
		super(Property_prediction_deep, self).__init__()
		# Encoder Part

		# Feature Selector
		self.mask=nn.Parameter(Variable(torch.ones(orig_atom_fea_len), requires_grad=True).to(device))

		self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len, bias=False).to(device)
		self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len,
											  random_seed=random_seed, activation=activation)
									for _ in range(n_conv)])
		#Property Prediction Part
		self.conv_to_fc1 = nn.Linear(h_fea_len, h_fea_len).to(device)
		self.activation1 = getActivation(activation)

		self.conv_to_fc2 = nn.Linear(h_fea_len, h_fea_len).to(device)
		self.activation2 = getActivation(activation)

		self.fc_out = nn.Linear(h_fea_len, 1).to(device)

	def forward(self, atom_fea,nbr_fea, nbr_fea_idx,crystal_atom_idx):
		# Encoder Part


		# # Feature Selector
		N = atom_fea.shape[0]
		mask = self.mask.repeat(N, 1)
		atom_fea = atom_fea * mask  #Element wise Multiplication
		atom_mask_feature=atom_fea.clone()

		# # Embedding Layer (92 -> 64)
		atom_fea = self.embedding(atom_fea)

		# # Convolution Layers
		for conv_func in self.convs:
			atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)

		# Property Prediction Part
		bt_atom_fea = [atom_fea[idx_map] for idx_map in crystal_atom_idx]
		prop_list = []
		for atom_fea in bt_atom_fea:
			atom_fea = F.normalize(atom_fea, dim=1, p=2)
			atom_fea = torch.mean(atom_fea, dim=0, keepdim=True)
			atom_fea = self.conv_to_fc1(atom_fea)
			atom_fea = self.activation1(atom_fea)
			atom_fea = self.conv_to_fc2(atom_fea)
			atom_fea = self.activation2(atom_fea)
			out = self.fc_out(atom_fea)
			prop_list.append(out)
		return prop_list,atom_mask_feature




