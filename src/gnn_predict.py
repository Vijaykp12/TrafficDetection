import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

data = torch.load("../data/processed/road_graph.pt")

class TrafficGNN(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = GCNConv(5,16)
        self.conv2 = GCNConv(16,3)

    def forward(self,data):

        x,edge_index = data.x,data.edge_index

        x = self.conv1(x,edge_index)
        x = F.relu(x)

        x = self.conv2(x,edge_index)

        return x


model = TrafficGNN()

model.load_state_dict(torch.load("../models/gnn_model.pth"))

model.eval()

with torch.no_grad():

    out = model(data)

pred = out.argmax(dim=1)

print(pred[:20])
