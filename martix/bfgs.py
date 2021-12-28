import numpy
from matplotlib import pyplot as plt


# 目标函数0阶信息
def func(x1, x2):
    funcVal = 5 * x1 ** 2 + 2 * x2 ** 2 + 3 * x1 - 10 * x2 + 4
    return funcVal


def grad(x1, x2):
    gradVal = numpy.array([[10 * x1 + 3], [4 * x2 - 10]])
    return gradVal


class BGFS(object):
    def __init__(self, seed=None, epslion=1.e-6, maxIter=300):
        self.__seed = seed
        self.__epsilon = epslion
        self.__MaxIter = maxIter
        self.__xPath = list()
        self.__fPath = list()

    def solve(self):
        1



