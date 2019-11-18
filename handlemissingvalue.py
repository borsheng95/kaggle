from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

#function comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

#approach 1 : Drop column
# 1st col is col name, 2rd col is index
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

reduced_X_train= X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing value):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

#approach 2: imputation (add mean value)

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
#fit means calculate imputer?
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

#Imputation removed columns name; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE from Approach 2 (imputation)")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

#approach 3: add axtension to Imputation

X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()
#col is column name at here
#generate a true false list for [col] new columns
for col in cols_with_missing:
    X_train_plus[col + 'was missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + 'was missing'] = X_valid_plus[col].isnull()

#Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("MAE from Approach 3 (An Extension to Imputation:)")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))
