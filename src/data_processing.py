import pandas as pd

# -----------------------------
# LOAD DATA
# -----------------------------

traffic = pd.read_csv("../data/raw/traffic.csv")

weather = pd.read_csv(
    "../data/raw/open-meteo-13.11N80.25E12m.csv",
    skiprows=3
)

holiday = pd.read_csv("../data/raw/holidays.csv")


# -----------------------------
# CLEAN TRAFFIC DATA
# -----------------------------

traffic["date_time"] = pd.to_datetime(traffic["date_time"])

traffic["hour"] = traffic["date_time"].dt.hour
traffic["day"] = traffic["date_time"].dt.dayofweek
traffic["month"] = traffic["date_time"].dt.month

traffic["date"] = traffic["date_time"].dt.date
traffic["date"] = pd.to_datetime(traffic["date"])

traffic["is_weekend"] = traffic["day"].apply(lambda x: 1 if x >= 5 else 0)


# -----------------------------
# CLEAN WEATHER DATA
# -----------------------------

weather.columns = weather.columns.str.strip()

weather = weather.rename(columns={
    "temperature_2m (°C)": "temperature",
    "precipitation (mm)": "precipitation",
    "cloudcover (%)": "cloudcover",
    "windspeed_10m (km/h)": "windspeed"
})

weather["time"] = pd.to_datetime(weather["time"])
weather["hour"] = weather["time"].dt.hour

weather_hourly = weather.groupby("hour").agg({
    "temperature":"mean",
    "precipitation":"mean",
    "cloudcover":"mean",
    "windspeed":"mean"
}).reset_index()


# -----------------------------
# MERGE TRAFFIC + WEATHER
# -----------------------------

merged = pd.merge(
    traffic,
    weather_hourly,
    on="hour",
    how="left"
)


# -----------------------------
# ADD HOLIDAY DATA
# -----------------------------

holiday["date"] = pd.to_datetime(holiday["date"])

merged = pd.merge(
    merged,
    holiday,
    on="date",
    how="left"
)

if "holiday" in merged.columns:
    merged["is_holiday"] = merged["holiday"].notna().astype(int)
else:
    merged["is_holiday"] = 0


# -----------------------------
# TRAFFIC LAG FEATURES
# -----------------------------

merged["traffic_lag1"] = merged["traffic_volume"].shift(1)
merged["traffic_lag2"] = merged["traffic_volume"].shift(2)
merged["traffic_lag3"] = merged["traffic_volume"].shift(3)

merged["traffic_lag1"].fillna(merged["traffic_volume"].mean(), inplace=True)
merged["traffic_lag2"].fillna(merged["traffic_volume"].mean(), inplace=True)
merged["traffic_lag3"].fillna(merged["traffic_volume"].mean(), inplace=True)


# -----------------------------
# CREATE TRAFFIC LEVEL
# -----------------------------

low = merged["traffic_volume"].quantile(0.33)
high = merged["traffic_volume"].quantile(0.66)

def classify(v):

    if v < low:
        return 0
    elif v < high:
        return 1
    else:
        return 2

merged["traffic_level"] = merged["traffic_volume"].apply(classify)


# -----------------------------
# FINAL DATASET
# -----------------------------

final_dataset = merged[[
"hour",
"day",
"month",
"is_weekend",
"temperature",
"precipitation",
"cloudcover",
"windspeed",
"is_holiday",
"traffic_lag1",
"traffic_lag2",
"traffic_lag3",
"traffic_level"
]]

final_dataset.to_csv(
"../data/processed/merged_dataset.csv",
index=False
)

print("Dataset processed successfully")
print(final_dataset.head())