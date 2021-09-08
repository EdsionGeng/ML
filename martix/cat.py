import numpy as np
import matplotlib.pyplot as plt
import h5py
#import skimage.transform as tf\
import sklearn
import tensorflow.compat.v1 as tf


def load_dataset():
    train_dataset = h5py.File(r'C:\Users\gengyongchang\Downloads\1第一个人工智能程序\datasets\train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])
    test_dataset = h5py.File(r'C:\Users\gengyongchang\Downloads\1第一个人工智能程序\datasets\test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])
    classes = np.array(test_dataset["list_classes"][:])  # 加载标签内别数据

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))  # 变换数组维度 （209,)变成（1,209）
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))  # 从(50)变成（1,50）
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

index = 30
plt.imshow(train_set_x_orig[index])
print("标签为" + str(train_set_y[:, index]) + ", 这是一个'" + classes[np.squeeze(train_set_y[:, index])].decode(
    "utf-8") + "' 图片.")
# 标签为[0], 这是一个'non-cat'图片.

print("train_set_x_orig shape: " + str(train_set_x_orig.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x_orig shape: " + str(test_set_x_orig.shape))
print("test_set_y shape: " + str(test_set_y.shape))
# train_set_x_orig shape: (209, 64, 64, 3)
# train_set_y shape: (1, 209) test_set_x_orig
# shape: (50, 64, 64, 3) test_set_y shape: (1, 50)
# 上面train_set_x_orig的各维度的含义分别是(样本数，图片宽，图片长，3个RGB通道)


m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = test_set_x_orig.shape[1]

# 为了方便后面进行矩阵运算，我们需要将样本数据进行扁平化和转置
# 处理后的数组各维度的含义是（图片数据，样本数）
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
# train_set_x_flatten shape: (12288, 209) test_set_x_flatten shape: (12288, 50)

# 下面我们对特征数据进行了简单的标准化处理（除以255，使所有值都在[0，1]范围内）
# 为什么要对数据进行标准化处理呢？简单来说就是为了方便后面进行计算，详情以后再给大家解释
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255
# 上面我们已经加载了数据，并且对数据进行了预处理，使其便于进行后面的运算。


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))  # w权重数组
    b = 0  # 偏置bias
    return w, b


def propagate(w, b, X, Y):
    """ 参数: w -- 权重数组，维度是(12288, 1)
     b -- 偏置bias X -- 图片的特征数据，维度是 (12288, 209) Y -- 图片对应的标签，0或1，
     维度是(1,209)​返回值: cost -- 成本 dw -- w的梯度 db -- b的梯度
    """
    m = X.shape[1]
    # 前向传播
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
    # 反向传播
    dZ = A - Y
    dw = np.dot(X, dZ.T) / m
    db = np.sum(dZ) / m
    # 将dw和db保存到字典里面
    grads = {"dw": dw,
             "db": db}
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    参数: w -- 权重数组，维度是 (12288, 1) b -- 偏置bias X -- 图片的特征数据，维度是 (12288, 209) Y -- 图片对应的标签，0或1，维度是(1,209)
    num_iterations -- 指定要优化多少次 learning_rate -- 学习步进，是我们用来控制优化步进的参数 print_cost -- 为True时，每优化100次就把成本cost打印出来,以便我们观察成本的变化

    返回值:
    params -- 优化后的w和b
    costs -- 每优化100次，将成本记录下来，成本越小，表示参数越优化
    """
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        # 计算得出梯度和成本
        dw = grads["dw"]
        db = grads["db"]
        # 进行梯度下降，更新参数，使其越来越优化，使成本越来越小
        w = w - learning_rate * dw
        b = b - learning_rate * db

    if i % 100 == 0:
        costs.append(cost)
        if print_cost:
            print("优化%i次后成本是: %f" % (i, cost))

    params = {"w": w, "b": b}
    return params, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        if A[0, i] >= 0.5:
            Y_prediction[0, i] = 1
    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    """

    # 初始化待训练的参数
    w, b = initialize_with_zeros(X_train.shape[0])

    # 使用训练数据来训练/优化参数
    parameters, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # 从字典中分别取出训练好的w和b
    w = parameters["w"]
    b = parameters["b"]

    # 使用训练好的w和b来分别对训练图片和测试图片进行预测
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    # 打印出预测的准确率
    print("对训练图片的预测准确率为: {}%".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("对测试图片的预测准确率为: {}%".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    return d


d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)
