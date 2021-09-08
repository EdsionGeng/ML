import numpy as np
import pandas as pd
import xgboost as xgb
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

train_x, train_y, test_x = pd.read_csv()
X, val_X, Y, val_Y = train_test_split(train_x, train_y, test_size=0.01, random_state=1, stratify=train_y)
xgb_val = xgb.DMatrix(val_X, label=val_Y)
xgb_train = xgb.DMatrix(X, label=Y)
xgb_test = xgb.DMatrix(test_x)

params = {'booster': 'gbtree',
          'objective': 'binary:logistic',
          'eval_metric': 'logloss',
          'gamma': 0.1,
          'max_depth': 8,
          'alpha': 0,
          'lambda': 10,
          'subsample': 0.7,
          'colsample_bytree': 0.5,
          'min_child_weight': 3,
          'silent': 0,
          'eta': 0.03,
          'seed': 1000,
          'nthread': -1,
          'missing': 1,
          'scale_pos_weight': (np.sum(Y == 0) / np.sum(Y == 1))  # 处理正负样本不均衡的问题

          }
plst = list(params.items())
num_rounds = 2000
watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]

# cross validation
result = xgb.cv(plst, xgb_train, num_boost_round=200, nfold=4, early_stopping_rounds=200, verbose_eval=True,
                folds=StratifiedKFold(n_splits=4).split(X, Y))

# train model and save
model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=200)
model.save_model('../data/model/xgb.model')
preds = model.predict(xgb_test)

threshold = 0.5
for pred in preds:
    result = 1 if pred > 0.5 else 0
