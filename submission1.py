import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split


train = pd.read_csv('train.csv')
print('data loaded successfully')
for col in train.columns:
    if 'cat' in col:
        train[col] = train[col].factorize()[0]

print('data preprocessing complete')
        
y = train['target']
X = train.drop('target', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print('train_test_split: complete')
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)
param = {'num_leaves': 31, 'objective': 'regression'}
num_round = 10
print('training model')
bst = lgb.train(param, train_data, num_round, valid_sets=[test_data])

test = pd.read_csv('test.csv')

for col in test.columns:
    if 'cat' in col:
        test[col] = test[col].factorize()[0]

print('predicting on test set')
preds = bst.predict(test, num_iteration = bst.best_iteration)
submission = pd.read_csv('sample_submission.csv')
submission['target'] = preds
print('saving submission file')
submission.to_csv('submission1.csv', index=False)
print('successfully saved submission file')
