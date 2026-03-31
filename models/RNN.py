import torch.nn as nn


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn     = nn.RNN(input_size, hidden_size)
        self.h2o     = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.softmax(self.h2o(h[0]))


class CharLSTM(nn.Module):
    """
    Same as CharRNN but uses an LSTM cell, which adds a cell state and
    three gates (input / forget / output) for better long-range memory.
    nn.LSTM returns (output, (h_n, c_n)); we use only h_n for classification.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size)
        self.h2o     = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.softmax(self.h2o(h[0]))