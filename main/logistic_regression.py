import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df_train = pd.DataFrame()
df_test = pd.DataFrame()
y_train = df_train['label'].values

ss = StandardScaler()

enc = OneHotEncoder()
feats = ['creativeID', 'adID', 'campaignID']

for i, feat in enumerate(feats):
    x_train = enc.fit_transform(df_train[feat].values.reshape(-1, 1))
    x_test = enc.fit_transform(df_test[feat].values.reshape(-1, 1))
    if i == 0:
        X_train, X_test = x_train, x_test
    else:
        X_Train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))

feats = ['price', 'age']
x_train = ss.fit_transform(df_train[feats].values)
x_test = ss.fit_transform(df_test[feats].values)
X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))

lr = LogisticRegression()
lr.fit(X_train, y_train)
proba_test = lr.predict_proba(X_test)[:, 1]
