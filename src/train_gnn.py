import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree
import numpy as np

# -----------------------------
# 1. LOAD DATA
# -----------------------------
try:
    data = torch.load("../data/processed/road_graph.pt", weights_only=False)
    num_nodes = data.num_nodes
    print(f"Successfully loaded graph with {num_nodes} nodes.")
except Exception as e:
    print(f"Error loading graph: {e}")
    exit()

# -----------------------------
# 2. FEATURE ENGINEERING & BALANCED LABELING
# -----------------------------
# Calculate normalized degree
deg = degree(data.edge_index[0], num_nodes=num_nodes)
# Square the degree to make intersections (3, 4+) stand out from roads (2)
degree_boosted = (deg**2) / (deg**2).max()

# Create a spatial gradient (simulates distance from city center)
spatial_bias = torch.linspace(0, 1, num_nodes)

# Constants (Normalized 0-1)
hour_norm = torch.full((num_nodes,), 9 / 24)
temp_norm = torch.full((num_nodes,), 30 / 50)

# FEATURE MATRIX (X)
data.x = torch.stack([degree_boosted, spatial_bias, hour_norm, temp_norm], dim=1).float()

# HEURISTIC LOGIC FOR LABELS
# We combine structural (degree) and spatial (location) factors
logic_score = (degree_boosted * 0.6) + (spatial_bias * 0.4)

# FORCE 33% Split for Low, Med, High using Quantiles
q1 = torch.quantile(logic_score, 0.33)
q2 = torch.quantile(logic_score, 0.66)

labels = torch.zeros(num_nodes, dtype=torch.long)
labels[logic_score >= q1] = 1  # Medium
labels[logic_score >= q2] = 2  # High
data.y = labels

# -----------------------------
# 3. DIAGNOSTICS
# -----------------------------
print("\n--- DATA DIAGNOSTICS ---")
print(f"Unique degree values: {torch.unique(deg).tolist()}")
unq, counts = torch.unique(data.y, return_counts=True)
for u, c in zip(unq, counts):
    label_name = ["Low", "Med", "High"][u.item()]
    print(f"Target {label_name} (Class {u.item()}): {c.item()} nodes")
print("------------------------\n")

# -----------------------------
# 4. GNN ARCHITECTURE (With Skip Connection)
# -----------------------------
class TrafficGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # Skip connection layer to keep original features alive
        self.lin_skip = torch.nn.Linear(input_dim, output_dim)
        self.final_lin = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Layer 1
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        
        # Layer 2
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        
        # Combine Graph Info + Original Node Info (Prevents Med:0)
        graph_out = self.final_lin(h)
        skip_out = self.lin_skip(x)
        
        return graph_out + skip_out

model = TrafficGNN(input_dim=4, hidden_dim=32, output_dim=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# -----------------------------
# 5. TRAINING LOOP
# -----------------------------
model.train()
for epoch in range(201):
    optimizer.zero_grad()
    out = model(data)
    
    # Weighted Loss: Encourage model to get Medium (Class 1) right
    weights = torch.tensor([1.0, 1.5, 1.2]).float() 
    loss = F.cross_entropy(out, data.y, weight=weights)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        preds = out.argmax(dim=1)
        l = (preds == 0).sum().item()
        m = (preds == 1).sum().item()
        h = (preds == 2).sum().item()
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Preds -> L: {l}, M: {m}, H: {h}")

# -----------------------------
# 6. SAVE OUTPUTS
# -----------------------------
model.eval()
with torch.no_grad():
    final_out = model(data)
    final_pred = final_out.argmax(dim=1)

torch.save(model.state_dict(), "../models/gnn_model.pth")
torch.save(final_pred, "../models/gnn_predictions.pt")
print("\nSuccess! Model and predictions saved.")