import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
import numpy as np

train = pd.read_csv('train.csv')
print('data loaded successfully')
for col in train.columns:
    if 'cat' in col:
        train[col] = train[col].factorize()[0]

test = pd.read_csv('test.csv')

for col in test.columns:
    if 'cat' in col:
        test[col] = test[col].factorize()[0]

print('data preprocessing complete')

features = ['id', 'cat0', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9', 'cont0', 'cont1', 'cont2', 'cont3', 'cont4', 'cont5', 'cont6', 'cont7', 'cont8', 'cont9', 'cont10', 'cont11', 'cont12', 'cont13']

# y = train['target']
# X = train.drop('target', axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=train['cat1'])
# print('train_test_split: complete')
# train_data = lgb.Dataset(X_train, label=y_train)
# test_data = lgb.Dataset(X_test, label=y_test)

def create_stratified_folds_for_regression(data_df, n_splits=5):
    """
    @param data_df: training data to split in Stratified K Folds for a continous target value
    @param n_splits: number of splits
    @return: the training data with a column with kfold id
    """
    data_df['kfold'] = -1
    # randomize the data
    data_df = data_df.sample(frac=1).reset_index(drop=True)
    # calculate the optimal number of bins based on log2(data_df.shape[0])
    num_bins = np.int(np.floor(1 + np.log2(len(data_df))))
    print(f"Num bins: {num_bins}")
    # bins value will be the equivalent of class value of target feature used by StratifiedKFold to 
    # distribute evenly the classed over each fold
    data_df.loc[:, "bins"] = pd.cut(pd.to_numeric(data_df['target'], downcast="signed"), bins=num_bins, labels=False)
    kf = model_selection.StratifiedKFold(n_splits=n_splits)
    
    # set the fold id as a new column in the train data
    for f, (t_, v_) in enumerate(kf.split(X=data_df, y=data_df.bins.values)):
        data_df.loc[v_, 'kfold'] = f
    
    # drop the bins column (no longer needed)
    data_df = data_df.drop("bins", axis=1)
    
    return data_df

n_splits = 5
print('Creating a Stratified fold')
train = create_stratified_folds_for_regression(train, n_splits)

def kfold_splits(n_splits, train_df):
    """
    Returns a collection of (fold, train indexes, validation indexes)
    @param n_splits: number of splits
    @param train_df: training data
    @return: a collection of (fold, train indexes, validation indexes)
    """
    all_folds = list(range(0, n_splits))
    kf_splits = []
    for fold in range(0, n_splits):
        train_folds = [x for x in all_folds if x != fold]
        trn_idx = train_df[train_df.kfold!=fold].index
        val_idx = train_df[train_df.kfold==fold].index
        kf_splits.append((fold, trn_idx, val_idx))
    return kf_splits

y = train['target']
oof = np.zeros(len(train))
predictions = np.zeros(len(test))

param = {'objective': 'regression', 'metric': 'rmse'}
# num_round = 10
# bst = lgb.train(param, train_data, num_round, valid_sets=[test_data])
# bst = lgb.train(param, train_data, valid_sets=[test_data])

# y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
# print('Validation Score is:', mean_squared_error(y_test, y_pred, squared=False))

print('training model')

for fold, trn_idx, val_idx in kfold_splits(n_splits, train):
    print(f"fold: {fold}, train len: {len(trn_idx)}, val len: {len(val_idx)}")
    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=y.iloc[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx][features], label=y.iloc[val_idx])
    bst = lgb.train(param, trn_data, valid_sets = [trn_data, val_data])
    oof[val_idx] = bst.predict(train.iloc[val_idx][features], num_iteration=bst.best_iteration)
    predictions += bst.predict(test[features], num_iteration=bst.best_iteration)/n_splits

print(f'CV score: {np.round(mean_squared_error(y, oof, squared=False),5)}')

# print('predicting on test set')
# preds = bst.predict(test, num_iteration = bst.best_iteration)
submission = pd.read_csv('sample_submission.csv')
submission['target'] = predictions
print('saving submission file')
submission.to_csv('submission4.csv', index=False)
print('successfully saved submission file')
