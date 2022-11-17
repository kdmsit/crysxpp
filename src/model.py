from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()

        #Feature Selector (Added by Authors of CrysXPP)
        self.mask = nn.Parameter(torch.ones(orig_atom_fea_len), requires_grad=True)

        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """

        # # Feature Selector (Added by Authors of CrysXPP)
        N = atom_fea.shape[0]
        mask = self.mask.repeat(N, 1)
        atom_fea = atom_fea * mask  # Element wise Multiplication
        atom_mask_feature = atom_fea.clone()


        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out,atom_mask_feature

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)


class CrystalAE(nn.Module):
	"""
	Create a crystal graph auto encoder to learn node representations through unsupervised training
	"""

	def __init__(self, orig_atom_fea_len, nbr_fea_len,atom_fea_len=64, n_conv=3,activation='softplus',random_seed=None):
		randomSeed(random_seed)
		super(CrystalAE, self).__init__()
		self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len,bias=False).to(device)
		self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,nbr_fea_len=nbr_fea_len) for _ in range(n_conv)])
		self.fc_adj = nn.Bilinear(atom_fea_len,atom_fea_len, 6).to(device)
		self.fc1 = nn.Linear(6, 6).to(device)

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

			# Atom Feature Reconstruction
			atom_feature_list.append(self.fc_atom_feature(atom_fea))
		return edge_prob_list, atom_feature_list
