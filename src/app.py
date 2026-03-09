import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree

from flask import Flask, request, jsonify
from flask_cors import CORS

import joblib
import pandas as pd
import requests
import math

app = Flask(__name__)
CORS(app)

# -------------------------
# GNN MODEL
# -------------------------

class TrafficGNN(torch.nn.Module):

    def __init__(self,input_dim,hidden_dim,output_dim):

        super().__init__()

        self.conv1 = GCNConv(input_dim,hidden_dim)
        self.conv2 = GCNConv(hidden_dim,hidden_dim)

        self.lin_skip = torch.nn.Linear(input_dim,output_dim)
        self.final_lin = torch.nn.Linear(hidden_dim,output_dim)

    def forward(self,data):

        x,edge_index = data.x,data.edge_index

        h = F.relu(self.conv1(x,edge_index))
        h = F.relu(self.conv2(h,edge_index))

        return self.final_lin(h) + self.lin_skip(x)

# -------------------------
# LOAD MODELS
# -------------------------

ml_model = joblib.load("../models/traffic_model.pkl")

gnn_model = TrafficGNN(4,32,3)

gnn_model.load_state_dict(
    torch.load("../models/gnn_model.pth",weights_only=True)
)

gnn_model.eval()

# -------------------------
# LOAD GRAPH
# -------------------------

base_graph = torch.load(
    "../data/processed/road_graph.pt",
    weights_only=False
)

num_nodes = base_graph.num_nodes

node_degrees = degree(
    base_graph.edge_index[0],
    num_nodes=num_nodes
)

# -------------------------
# WEATHER API
# -------------------------

def get_weather():

    try:

        url = "https://api.open-meteo.com/v1/forecast?latitude=13.0827&longitude=80.2707&current_weather=true"

        r = requests.get(url,timeout=2).json()

        return r["current_weather"]

    except:

        return {
            "temperature":30,
            "windspeed":10
        }

# -------------------------
# RUSH HOUR MODEL
# -------------------------

def rush_hour(hour):

    morning = math.exp(-((hour-9)**2)/10)
    evening = math.exp(-((hour-18)**2)/10)

    base = 0.25  # base traffic level for daytime

    return max(morning, evening, base)

# -------------------------
# PREDICT API
# -------------------------

@app.route("/predict",methods=["POST"])

def predict():

    data = request.json

    hour = int(data.get("hour",9))

    weather = get_weather()

    # ---------------------
    # TIME INTENSITY
    # ---------------------

    time_factor = rush_hour(hour)

    # ---------------------
    # ML CITY MODEL
    # ---------------------

    lag_base = 2000 + (time_factor*4000)

    features = pd.DataFrame([{

        "hour":hour,
        "day":1,
        "month":3,
        "is_weekend":0,

        "temperature":weather["temperature"],
        "precipitation":0,
        "cloudcover":50,
        "windspeed":weather["windspeed"],
        "is_holiday":0,

        "traffic_lag1":lag_base,
        "traffic_lag2":lag_base-500,
        "traffic_lag3":lag_base-1000

    }])

    ml_pred = ml_model.predict(features)[0]

    mapping = {
        0:"Low",
        1:"Medium",
        2:"High"
    }

    city_status = mapping[ml_pred]

    # ---------------------
    # SPATIAL ROAD SCORE
    # ---------------------

    # random traffic seeds across city
    seed = torch.rand(num_nodes)

    # stronger seeds during rush hour
    seed = seed * (0.5 + time_factor)

    # propagate traffic through network
    edge_index = base_graph.edge_index

    traffic = seed.clone()

    for _ in range(3):
        src = edge_index[0]
        dst = edge_index[1]
        traffic[dst] += traffic[src]*0.15

    # normalize
    traffic = (traffic - traffic.min()) / (traffic.max() - traffic.min())

    base_score = traffic

    # ---------------------
    # CLASSIFY ROADS
    # ---------------------

    q_high = torch.quantile(base_score,0.75)
    q_med = torch.quantile(base_score,0.45)

    road_preds = torch.zeros(num_nodes,dtype=torch.long)

    road_preds[base_score>q_med] = 1
    road_preds[base_score>q_high] = 2

    # ---------------------
    # APPLY TIME EFFECT
    # ---------------------

    if time_factor < 0.25:

        road_preds[:] = 0
        city_status = "Low"

    elif time_factor < 0.5:

        road_preds[road_preds == 2] = 1
        city_status = "Medium"

    elif time_factor > 0.8:

        road_preds[road_preds == 1] = 2
        city_status = "High"

    # ---------------------
    # DEBUG DISTRIBUTION
    # ---------------------

    unique,counts = torch.unique(
        road_preds,
        return_counts=True
    )

    print(
        f"Hour {hour} | Status {city_status} | Distribution:",
        dict(zip(unique.tolist(),counts.tolist()))
    )

    # ---------------------
    # RETURN RESULT
    # ---------------------

    return jsonify({

        "traffic":city_status,
        "road_predictions":road_preds.tolist(),
        "weather":weather

    })

# -------------------------
# RUN SERVER
# -------------------------

if __name__ == "__main__":

    app.run(
        port=5000,
        debug=True
    )