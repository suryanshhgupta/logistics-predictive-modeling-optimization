"""Predictive modeling in logistics using simulated data. Run: python logistics_predictive_model.py"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df=pd.read_csv(Path(__file__).with_name("logistics_prediction_dataset.csv"))
features=["distance_km","weight_kg","transport_mode","region","weather_condition","traffic_level","priority"]
X,y=df[features],df["delivery_days"]
numeric=["distance_km","weight_kg"]; categorical=[c for c in features if c not in numeric]
prep=ColumnTransformer([
 ("num",Pipeline([("imputer",SimpleImputer(strategy="median"))]),numeric),
 ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical)])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=21)
models={"Linear Regression":LinearRegression(),
 "Random Forest":RandomForestRegressor(n_estimators=160,max_depth=12,min_samples_leaf=3,random_state=21)}
for name,model in models.items():
 pipe=Pipeline([("preprocess",prep),("model",model)])
 pipe.fit(X_train,y_train); pred=pipe.predict(X_test)
 print(f"\\n{name}")
 print("MAE:",round(mean_absolute_error(y_test,pred),3),"days")
 print("RMSE:",round(np.sqrt(mean_squared_error(y_test,pred)),3),"days")
 print("R-squared:",round(r2_score(y_test,pred),3))
rf=Pipeline([("preprocess",prep),("model",RandomForestRegressor(n_estimators=160,max_depth=12,min_samples_leaf=3,random_state=21))])
cv=-cross_val_score(rf,X_train,y_train,scoring="neg_mean_absolute_error",cv=KFold(5,shuffle=True,random_state=21))
print(f"\\n5-fold CV MAE: {cv.mean():.3f} +/- {cv.std():.3f} days")
# Optimization ideas: flag high-risk deliveries, compare feasible modes under cost/capacity constraints,
# consolidate compatible shipments, and validate each change in a controlled pilot.
