import numpy as np
import pycorrector


class LSTM():
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers


array = [1, 0, 2, 3, 4, 6]
print(np.argmin(array))
corrected_sent, detail = pycorrector.correct('少先队员因该为老人让坐')
print(corrected_sent, detail)
