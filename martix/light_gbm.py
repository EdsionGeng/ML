from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

train_data = lgb.Dataset(X_train, label=y_train)
validation_data = lgb.Dataset(X_test, label=y_test)

params = {'learning_rate': 0.1, 'lambda_l1': 0.1, 'lambda_l2': 0.2,
          'max_depth': 4, 'objective': 'multiclass', 'num_class': 3}

gbm = lgb.train(params, train_data, valid_sets=[validation_data])
y_pred = gbm.predict(X_test)
y_pred = [list(x).index(max(x)) for x in y_pred]

print(y_pred)
print(accuracy_score(y_test, y_pred))
