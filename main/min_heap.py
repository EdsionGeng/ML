class MinHeap(object):
    def __init__(self):
        self.data = []
        self.count = 0

    def add(self, x):
        self.data.append(x)
        self.shiftUp(self.count)
        self.count += 1

    def pop(self):
        if self.count:
            ret = self.data[0]
            self.count -= 1
            self.data[0] = self.data[-1]
            self.shiftDown(0)
            return ret

    def shiftUp(self, index):
        parent = (index - 1) >> 1
        while index > 0 and self.data[index] < self.data[parent]:
            self.data[index], self.data[parent] = self.data[parent], self.data[index]
            index = parent
            parent = (index - 1) >> 1

    def shiftDown(self, index):
        min_child = (index << 1) + 1
        while min_child < self.count:
            if min_child + 1 < self.count and self.data[min_child + 1] < self.data[min_child]:
                min_child = min_child + 1
            if self.data[index] < self.data[min_child]:
                break
            self.data[index], self.data[min_child] = self.data[min_child], self.data[index]
            index = min_child
            min_child = (index << 1) + 1


if __name__ == '__main__':
    minHeap = MinHeap()
    minHeap.add(7)
    minHeap.add(6)
    minHeap.add(5)
    minHeap.add(11)
    minHeap.add(13)
    minHeap.add(4)
    minHeap.add(2)

    print(minHeap.pop())
