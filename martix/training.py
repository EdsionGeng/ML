import numpy as np
import pandas as pd


class MaskListener:
    a = {"a": 1, "b": 2, "c": 3}
    b = [1, 2, 3, 4]

    def __init__(self, in_features, out_features, bias=True):
        super(MaskListener, self).__init__(in_features, out_features, bias)
        pd.read_csv("")
        self.mask = None

    def set_mask(self, mask, kernel_size=7):
        a = np.array([[1, 6, 5, 2], [9, 6, 5, 9], [3, 7, 9, 1]])
        print(np.argmax(a, axis=0))  # 横着比较返回列号
        print(np.argmax(a, axis=1))  # 竖着比较返回行号
        self.mask = mask.detach()

    def main(self):
        a = {"key1": "a", "key2": "b"}
        for i,j in a.items():
            print(i,j)

    if __name__ == '__main__':
        main(self=None)
        print(a["b"])
        for i in b:
            if i > 2:
                print(i)
        print("log start")
