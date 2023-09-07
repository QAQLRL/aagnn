import dgl
import torch
g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]))
g.ndata['h'] = torch.ones(5, 2)
g.apply_edges(lambda edges: {'x' : edges.src['h'] + edges.dst['h']})
g.edata['x']