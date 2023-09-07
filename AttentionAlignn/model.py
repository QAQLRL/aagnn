from .layers import MLPLayer,ALIGNNConv,GAT_Crystal,GCNII
from .utils import RBFExpansion
import torch
from torch import nn
import dgl
from dgl.nn.pytorch.glob import AvgPooling
from dgl import DGLGraph
from dgl.nn.pytorch.conv import GATConv
from typing import Tuple,Union
import numpy as np
from .config import TrainConfig,ALIGNNConfig

class BondAngleGraphAttention(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self,config:ALIGNNConfig):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        self.classification = config.classification
        self.atom_embedding = MLPLayer(
            config.atom_input_features, config.hidden_features
        )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8.0, bins=config.edge_input_features,),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1, vmax=1.0, bins=config.triplet_input_features,
            ),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(config.hidden_features, config.hidden_features, config.hidden_features, config.heads)
                for idx in range(config.alignn_layers)
            ]
        )
        self.gat_layers = nn.ModuleList(
            [
                GAT_Crystal(
                    config.hidden_features, config.hidden_features,num_heads=config.heads
                )
                for idx in range(config.gat_layers)
            ]
        )
        layer_number = config.alignn_layers + config.gat_layers
        self.gcnii_layers =  [
               GCNII(
                    2*config.hidden_features, layer_number+idx
                )
                for idx in range(1,config.gcnii_layers+1)
            ]

        self.readout = AvgPooling()
        
        if self.classification:
            self.fc = nn.Linear(config.hidden_features, config.num_classes)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc = nn.Linear(config.hidden_features, config.output_features)
        self.link = None
        
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            self.fc.bias.data = torch.tensor(
                np.log(avg_gap), dtype=torch.float
            )
        elif config.link == "logit":
            self.link = torch.sigmoid

    def forward(
        self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]
    ):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        g, lg = g
        lg = lg.local_var()

          
        # angle features (fixed)
        z = self.angle_embedding(lg.edata.pop("h"))

        g = g.local_var()

        # initial node features: atom feature network...
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)

        # initial bond features
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        y = self.edge_embedding(bondlength)
        
        
        # x_clone = x.clone()
        # # gat updates: update node
        # for gat_layer in self.gat_layers:
        #     x_clone = gat_layer(g, x_clone)
        #     x_clone = torch.mean(x_clone,dim=1)
        
        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)
        
       
    
        # gcnii update: update note 
        # h_0 = x_cat
        # for gcnii_layer in self.gcnii_layers:
            
        #     x = gcnii_layer(g,x_cat,h_0)
        h = self.readout(g, x)
        out = self.fc(h)

        if self.link:
            out = self.link(out)
        if self.classification:
            # out = torch.round(torch.sigmoid(out))
            out = self.softmax(out)
        return torch.squeeze(out)
