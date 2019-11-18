import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

y = data.Price
#axis =0 mean row, axis =1 mean column
X= data.drop(['Price'], axis=1)
# set the random_state in same value to obtain same random number
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()]
# if inplace = true, do the operation inplace and return None
X_train_full.drop[cols_with_missing, axis=1, inplace=True]
X_valid_full.drop[cols_with_missing, axis=1, inplace=True]

#Cardinality means the number of unique values in a column
#Seleect categorical columns with relatively low cardinality (convenient but arbitray)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"]

#Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X.X_train_full[cname].dtype in ['int64', 'float64']]

#keep selected columns only
my_cols = low_cardinality_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

# Get list of categorical variables
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# Approach 1: Drop the categorical column
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

print("MAE from approach 1 (Drop Categorical variables)")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))

# Approach 2: Label encoding
from sklearn.preprocessing import LabelEncoder

#Make copy to avoid changing original data
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
for col in object_cols:
    label_X_train[col] = label_encoder.fit_transform(X_train[col])
    label_X_valid[col] = label_encoder.transform(X_valid[col])

print("MAE from approach 2 (Label encoding):")
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))

# Approach 3: one hot encoding
from sklearn.preprocessing import OneHotEncoder
#Apply one-Hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown = 'ignore', sparse = False)
OH_cols_train = pd.DataFrame(OH.encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DaaFrame(OH.OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

#Remove categorical columns and replace with one hot encoding
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

#pd.concat is combine two set of data
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

print("MAE form approach 3 (One-Hot Encoding)")
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
