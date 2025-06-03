import torch
import torch.nn as nn

class FastTextLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim=300, hidden_dim=128, output_dim=2, num_layers=1):
        super(FastTextLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        # h_n[-1] son katmandaki gizli durum
        out = self.fc(h_n[-1])
        return out