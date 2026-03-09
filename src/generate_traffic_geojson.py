import json
import torch
# We need to import Data to help torch recognize the object type
from torch_geometric.data import Data 

# 1. LOAD DATA
# Set weights_only=False because we are loading a custom Graph object, not just weights
data = torch.load("../data/processed/road_graph.pt", weights_only=False)

with open("../frontend/traffic-dashboard/public/export.geojson", encoding="utf-8") as f:
    geo = json.load(f)

# Load the predictions (these are usually just simple tensors/lists)
predictions = torch.load("../models/gnn_predictions.pt", weights_only=False).tolist()

# 2. MATCH COORDINATES TO PREDICTIONS
# PyTorch Geometric doesn't store coordinates in 'nodes'. 
# In your build_graph.py, you added points to G in a specific order.
# Let's recreate the unique coordinate list from the GeoJSON to match the graph indices.

unique_coords = []
seen = set()

for feature in geo["features"]:
    for p in feature["geometry"]["coordinates"]:
        p_tuple = tuple(p)
        if p_tuple not in seen:
            unique_coords.append(p_tuple)
            seen.add(p_tuple)

# Safety Check: Do our unique coordinates match the number of nodes in the GNN?
if len(unique_coords) != data.num_nodes:
    print(f"Warning: GeoJSON unique points ({len(unique_coords)}) " 
          f"mismatch GNN nodes ({data.num_nodes})")

# Create the mapping: coordinate tuple -> traffic level
coord_to_traffic = {coord: predictions[i] for i, coord in enumerate(unique_coords)}

# 3. UPDATE GEOJSON PROPERTIES
for feature in geo["features"]:
    coords = feature["geometry"]["coordinates"]
    
    segment_levels = []
    for p in coords:
        p_tuple = tuple(p)
        if p_tuple in coord_to_traffic:
            segment_levels.append(coord_to_traffic[p_tuple])
    
    if segment_levels:
        # Use the maximum traffic level found in any point of the segment 
        # (This makes congestion "pop" more on the map)
        avg_traffic = int(round(sum(segment_levels) / len(segment_levels)))
        feature["properties"]["congestion"] = avg_traffic
    else:
        feature["properties"]["congestion"] = 0

# 4. SAVE FOR FRONTEND
output_path = "../frontend/traffic-dashboard/public/traffic_live.json"
with open(output_path, "w") as f:
    json.dump(geo, f)

print(f"Successfully generated {output_path}")
print(f"Sample Property: {geo['features'][0]['properties']}")