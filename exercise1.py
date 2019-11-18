import os
if not os.path.exists("../input/train.csv"):
    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")
    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv")

import pandas as pd
from sklearn.model.selection import train_test_split

#Read the data
X_full = pd.read_csv('../input/train.csv')
X_test = pd.read_csv('../input/test.csv')

#obtain target and predictors
y = X_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stF1rSF', '2ndF1rsF', 'FullBath', 'BedroomAbGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
X_test = X_test[features].copy()

#Break off validation set from train data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)

#overview of data
X_train.head()

#generate model
from sklearn.ensemble import RandomForestRegressor

model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]

#train data
from sklearn.metrics import mean_absolute_error

def score_model(model, X_t=X_train, X_v=valid, y_t=y_train, y_v=y_train):
    model.fit(X_t, y_t)
    preds =model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(0, len(models)):
    mae = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))
