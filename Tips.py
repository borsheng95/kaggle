#Investigating Cardinality
#Get number of unique entries in each column with categorical Data
object_nunique = list(map(lambda col : X_train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

#print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x:x[1])

# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
