import lightgbm as lgb
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

print('Loading data...')
train_x, train_y, test_x = pd.read_csv('')

X, val_X, Y, val_Y = train_test_split(train_x, train_y, test_size=0.05, random_state=1, stratify=train_y)
X_train = X
y_train = Y
X_test = val_X
y_test = val_Y
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {'boosting_type': 'gbdt',
          'objective': 'binary',
          #'objective':'multiclass'
          #''metric:'multi_error'
          'metric': {'binary_loss', 'auc'},
          'num_leaves': 5,
          'max_depth': 6,
          'min_data_in_leaf': 450,
          'learning_rate': 0.1,
          'feature_fraction': 0.9,
          'bagging_fraction': 0.95,
          'bagging_freq': 5,
          'lambda_l1': 1,
          'lambda_l2': 0.001,
          'min_gain_to_split': 0.2,
          'verbose': 5,
          'is_unbalance': True
          }
# train
print('Start training...')
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=lgb_eval,
                early_stopping_rounds=500
                )
print('Start predicting...')
preds = gbm.predict(test_x, num_iteration=gbm.best_iteration)

threshold = 0.5
for pred in preds:
    result = 1 if pred > threshold else 0

importance = gbm.feature_importance()
names = gbm.feature_name()
with open('./feature_importance.txt', 'w+') as file:
    for index, im in enumerate(importance):
        string = names[index] + ',' + str(im) + '\n'
        file.write(string)
