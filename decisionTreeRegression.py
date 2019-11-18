import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model.selection import train_test_split

file_path = '../input/train.csv'

home_data = pd.read_csv(file_path)

y = home_data.SalePrice

features = ['LotArea', 'YearBuilt', '1stF1rSF', '2ndF1rSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

model = DesicionTreeRegressor(random_state=1)

model.fit(train_X, train_y)

val_predictions = model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

model = DesicionTreeRegressor(max_leaf_nodes=100, random_state=1)

model.fit(train_X, train_y)

val_predictions = model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best max_leaf_nodes: {:,.0f}".format(val_mae))

#inspect.getargspec(func) can check args

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
print("validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
