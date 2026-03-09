import json
import networkx as nx
import torch
from torch_geometric.utils import from_networkx

# -----------------------------
# LOAD ROAD NETWORK
# -----------------------------

with open("../frontend/traffic-dashboard/public/export.geojson", encoding="utf-8") as f:
    geo = json.load(f)

G = nx.Graph()

# -----------------------------
# ROAD IMPORTANCE MAPPING
# -----------------------------

road_importance = {

"motorway": 1.0,
"trunk": 0.9,
"primary": 0.8,
"secondary": 0.7,
"tertiary": 0.6,
"residential": 0.4,
"service": 0.3

}

# -----------------------------
# BUILD GRAPH
# -----------------------------

for feature in geo["features"]:

    coords = feature["geometry"]["coordinates"]

    road_type = feature["properties"].get("highway","residential")

    importance = road_importance.get(road_type,0.5)

    for i in range(len(coords)-1):

        p1 = tuple(coords[i])
        p2 = tuple(coords[i+1])

        G.add_edge(p1,p2,weight=importance)

print("Nodes:",G.number_of_nodes())
print("Edges:",G.number_of_edges())

# -----------------------------
# CONVERT TO PYTORCH GRAPH
# -----------------------------

data = from_networkx(G)

torch.save(data,"../data/processed/road_graph.pt")

print("Graph saved successfully")