import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree
from flask import Flask, request, jsonify
from flask_cors import CORS
import networkx as nx
import joblib
import pandas as pd
import requests
import json
import math
from scipy.spatial import KDTree

app = Flask(__name__)
CORS(app)

# -------------------------
# LOAD ROAD GEOJSON
# -------------------------
with open("../frontend/traffic-dashboard/public/export.geojson", encoding = "utf-8") as f:
    roads_geojson = json.load(f)

road_features = roads_geojson["features"]

# -------------------------
# GNN MODEL
# -------------------------
class TrafficGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin_skip = torch.nn.Linear(input_dim, output_dim)
        self.final_lin = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return self.final_lin(h) + self.lin_skip(x)

# -------------------------
# LOAD MODELS & GRAPH
# -------------------------
try:
    ml_model = joblib.load("../models/traffic_model.pkl")

    gnn_model = TrafficGNN(4, 32, 3)
    gnn_model.load_state_dict(torch.load("../models/gnn_model.pth", weights_only=True))
    gnn_model.eval()

    base_graph = torch.load("../data/processed/road_graph.pt", weights_only=False)

    print(base_graph)

    num_nodes = base_graph.num_nodes
    node_degrees = degree(base_graph.edge_index[0], num_nodes=num_nodes)

except Exception as e:
    print(f"Initialization Error: {e}")

# -------------------------
# BUILD ROUTING GRAPH FROM GEOJSON
# -------------------------

routing_graph = nx.Graph()

node_lookup = {}
node_coords = []
node_to_coord = {}

node_id = 0

for feature in road_features:

    coords = feature["geometry"]["coordinates"]

    for i in range(len(coords) - 1):

        start = tuple(coords[i])
        end = tuple(coords[i + 1])

        if start not in node_lookup:
            node_lookup[start] = node_id
            node_to_coord[node_id] = list(start)
            node_coords.append(list(start))
            node_id += 1

        if end not in node_lookup:
            node_lookup[end] = node_id
            node_to_coord[node_id] = list(end)
            node_coords.append(list(end))
            node_id += 1

        u = node_lookup[start]
        v = node_lookup[end]

        dist = math.dist(start, end)

        routing_graph.add_edge(u, v, weight=dist)

# Build KDTree for nearest node lookup
spatial_index = KDTree(node_coords)

# -------------------------
# HELPERS
# -------------------------

df_coords = pd.read_csv("../data/processed/chennai_roads_clean.csv")
# Build location list from CSV
# -------------------------
# BUILD LOCATION MAP FROM CSV
# -------------------------

location_map = {}

for _, row in df_coords.iterrows():

    name = str(row.get("road_name", "")).strip()

    if name == "" or name == "nan":
        continue

    try:
        lon = float(row["longitude"])
        lat = float(row["latitude"])
    except:
        continue

    if name not in location_map:
        location_map[name] = [lon, lat]

    # limit dropdown size for performance
    if len(location_map) > 200:
        break

@app.route("/locations", methods=["GET"])
def get_locations():
    return jsonify(dict(location_map))

def get_weather():
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=13.0827&longitude=80.2707&current_weather=true"
        r = requests.get(url, timeout=2).json()
        return r["current_weather"]
    except:
        return {"temperature": 19, "windspeed": 10}

def rush_hour(hour):
    morning = math.exp(-((hour - 9) ** 2) / 10)
    evening = math.exp(-((hour - 18) ** 2) / 10)
    return max(morning, evening, 0.25)

# -------------------------
# ENDPOINTS
# -------------------------

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    hour = int(data.get("hour", 9))

    weather = get_weather()
    time_factor = rush_hour(hour)

    lag_base = 2000 + (time_factor * 4000)

    features = pd.DataFrame([{
        "hour": hour,
        "day": 1,
        "month": 3,
        "is_weekend": 0,
        "temperature": weather["temperature"],
        "precipitation": 0,
        "cloudcover": 50,
        "windspeed": weather["windspeed"],
        "is_holiday": 0,
        "traffic_lag1": lag_base,
        "traffic_lag2": lag_base - 500,
        "traffic_lag3": lag_base - 1000
    }])

    ml_pred = ml_model.predict(features)[0]

    mapping = {0: "Low", 1: "Medium", 2: "High"}
    city_status = mapping[ml_pred]

    seed = torch.rand(num_nodes) * (0.5 + time_factor)
    traffic = seed.clone()

    edge_index = base_graph.edge_index

    for _ in range(3):
        traffic[edge_index[1]] += traffic[edge_index[0]] * 0.15

    traffic = (traffic - traffic.min()) / (traffic.max() - traffic.min())

    base_score = traffic

    q_high = torch.quantile(base_score, 0.75)
    q_med = torch.quantile(base_score, 0.45)

    road_preds = torch.zeros(num_nodes, dtype=torch.long)

    road_preds[base_score > q_med] = 1
    road_preds[base_score > q_high] = 2

    if time_factor < 0.25:
        road_preds[:] = 0
        city_status = "Low"

    elif time_factor > 0.8:
        city_status = "High"

    return jsonify({
        "traffic": city_status,
        "road_predictions": road_preds.tolist(),
        "weather": weather
    })

@app.route("/route_by_name", methods=["POST"])
def route_by_name():
    data = request.json
    start_coords = data.get("start_coords")
    end_coords = data.get("end_coords")
    hour = int(data.get("hour", 9))
    is_emergency = data.get("isEmergency", False) # Get emergency flag

    # 1. Find nearest nodes in the graph
    _, start_idx = spatial_index.query(start_coords)
    _, end_idx = spatial_index.query(end_coords)
    start_idx, end_idx = int(start_idx), int(end_idx)

    # 2. Reset and Apply Weights based on Traffic/Emergency
    intensity = rush_hour(hour)
    
    for u, v, d in routing_graph.edges(data=True):
        # Base weight is the physical distance
        base_dist = math.dist(node_to_coord[u], node_to_coord[v])
        
        if is_emergency:
            # Emergency Mode: "Green Wave" (All roads are fast/cheap)
            routing_graph[u][v]["weight"] = base_dist * 0.1 
        else:
            # Normal Mode: Apply Traffic Bias
            bias = 1.0
            if intensity > 0.7: bias = 15.0
            elif intensity > 0.4: bias = 4.0
            routing_graph[u][v]["weight"] = base_dist * bias

    try:
        # --- PATH 1: MAIN OPTIMIZED ROUTE ---
        main_path = nx.shortest_path(routing_graph, source=start_idx, target=end_idx, weight="weight")
        main_coords = [node_to_coord[p] for p in main_path]
        
        # --- PATH 2: ALTERNATIVE ROUTE ---
        # Temporarily penalize edges used in the main path to force a different route
        original_weights = {}
        for i in range(len(main_path) - 1):
            u, v = main_path[i], main_path[i+1]
            original_weights[(u, v)] = routing_graph[u][v]["weight"]
            routing_graph[u][v]["weight"] *= 50.0 # Make these roads "expensive"

        try:
            alt_path = nx.shortest_path(routing_graph, source=start_idx, target=end_idx, weight="weight")
            alt_coords = [node_to_coord[p] for p in alt_path]
        except:
            alt_coords = [] # Fallback if no alternative exists

        # Restore weights so future requests aren't broken
        for (u, v), weight in original_weights.items():
            routing_graph[u][v]["weight"] = weight

        return jsonify({
            "coordinates": main_coords,
            "alt_coordinates": alt_coords,
            "is_emergency": is_emergency
        })

    except nx.NetworkXNoPath:
        return jsonify({"error": "No path found"}), 404
# -------------------------
# RUN SERVER
# -------------------------

if __name__ == "__main__":
    app.run(port=5000, debug=True)
