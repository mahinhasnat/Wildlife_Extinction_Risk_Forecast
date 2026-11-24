"""
REAL species-level ML (needs raw data that must be downloaded by you).
Input CSV  species_raw.csv  (one row per species) columns:
  name, group, status_2000, status_2015, habitat_forest(0/1), habitat_wetland(0/1),
  habitat_coastal(0/1), body_mass_kg, range_km2, hunting_trade(0/1), pop_trend(-1/0/1),
  protected_area_overlap(0..1)
Where to get it: IUCN Red List API (token, free for research), Bangladesh Red List PDFs
(portals.iucn.org RL-549.3-00x), Forest Dept lists, GFW habitat loss by district.
Target: status_2015 worse than status_2000 (deterioration) -> then predict 2030 risk.
Model: gradient boosting with stratified CV; reports AUC and feature importance.
"""
import sys, numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
ORDER = {"LC": 0, "NT": 1, "VU": 2, "EN": 3, "CR": 4, "RE": 5}
df = pd.read_csv(sys.argv[1])
df["y"] = (df.status_2015.map(ORDER) > df.status_2000.map(ORDER)).astype(int)
X = pd.get_dummies(df.drop(columns=["name", "status_2000", "status_2015", "y"]), columns=["group"])
m = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=.05, random_state=0)
p = cross_val_predict(m, X, df.y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
print("CV AUC:", round(roc_auc_score(df.y, p), 3), "n=", len(df), "positives=", df.y.sum())
m.fit(X, df.y)
df["risk_score"] = m.predict_proba(X)[:, 1]
print(pd.Series(m.feature_importances_, X.columns).sort_values(ascending=False).head(10))
df.sort_values("risk_score", ascending=False)[["name", "group", "status_2015", "risk_score"]].to_csv("species_risk_ranking.csv", index=False)
