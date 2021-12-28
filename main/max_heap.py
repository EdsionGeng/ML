class MaxHeap(object):
    def __init__(self):
        self._data = []
        self._count = 0

    def size(self):
        return self._count

    def isEmpty(self):
        return self._count == 0

    def add(self, x):
        self._data.append(x)
        self.shift_up(self._count)
        self._count += 1

    def pop(self):
        ret = self._data[0]
        self._count -= 1
        self._data[0] = self._data[-1]
        self.shift_down(0)
        return ret

    def shift_up(self, index):
        parent = (index - 1) >> 1
        while index > 0 and self._data[index] > self._data[parent]:
            self._data[index], self._data[parent] = self._data[parent], self._data[index]
            index = parent
            parent = (index - 1) >> 1

    def shift_down(self, index):
        max_child = (index << 1) + 1
        while max_child < self._count:
            if max_child + 1 < self._count and self._data[max_child + 1] > self._data[max_child]:
                max_child = max_child + 1
            if self._data[index] < self._data[max_child]:
                self._data[index], self._data[max_child] = self._data[max_child], self._data[index]
                index = max_child
                max_child = (index << 1) + 1

            else:
                break


if __name__ == '__main__':
    maxHeap = MaxHeap()
    maxHeap.add(15)
    maxHeap.add(10)
    maxHeap.add(12)
    maxHeap.add(5)
    maxHeap.add(9)
    maxHeap.add(8)
    maxHeap.add(7)
    maxHeap.add(11)
    maxHeap.pop()
