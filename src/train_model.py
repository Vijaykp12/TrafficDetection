import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# -----------------------
# LOAD DATA
# -----------------------

data = pd.read_csv("../data/processed/merged_dataset.csv")

X = data.drop("traffic_level", axis=1)
y = data["traffic_level"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# -----------------------
# RANDOM FOREST MODEL
# -----------------------

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train,y_train)

rf_pred = rf.predict(X_test)

print("\nRandomForest Accuracy:",accuracy_score(y_test,rf_pred))

print("\nRandomForest Report")
print(classification_report(y_test,rf_pred))


# -----------------------
# XGBOOST MODEL
# -----------------------

xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6
)

xgb.fit(X_train,y_train)

xgb_pred = xgb.predict(X_test)

print("\nXGBoost Accuracy:",accuracy_score(y_test,xgb_pred))

print("\nXGBoost Report")
print(classification_report(y_test,xgb_pred))


# -----------------------
# SAVE BEST MODEL
# -----------------------

if accuracy_score(y_test,xgb_pred) > accuracy_score(y_test,rf_pred):

    print("\nSaving XGBoost model")

    joblib.dump(xgb,"../models/traffic_model.pkl")

else:

    print("\nSaving RandomForest model")

    joblib.dump(rf,"../models/traffic_model.pkl")