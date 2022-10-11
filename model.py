##
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
import torchvision
##
batch_size = 16
img_width = 200
img_height = 50


class CaptchaModel(nn.Module):
    def __init__(self, vocab_size, dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding="same")
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding="same")
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        new_shape = (img_height // 4) * 64
        print("======== new_shape =========", new_shape)
        self.fc1 = nn.Linear(in_features=768, out_features=64)
        self.dropout = nn.Dropout(dropout)
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=128, bidirectional=True, batch_first=True)
        self.lstm_dropout = nn.Dropout(0.25)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=64, bidirectional=True, batch_first=True)
        self.logits = nn.Linear(128, vocab_size)

    def forward(self, input):
        x = self.conv1(input)
#        print("======== conv1 =========", x.shape)
        x = self.relu1(x)
        x = self.pool1(x)
#        print("======== pool1 =========", x.shape)
        x = self.conv2(x)
#        print("======== conv2 =========", x.shape)
        x = self.relu2(x)
        x = self.pool2(x)
#        print("======== pool2 =========", x.shape)
        x = x.view(x.shape[0],  -1, x.shape[3])
        x = x.permute(0, 2, 1)
#        print("======== view =========", x.shape)
        x = self.fc1(x)
#        print("======== fc1 =========", x.shape)
        x = self.dropout(x)
        x, _ = self.lstm1(x)
#        print("======== lstm1 =========", x.shape)
        x = self.lstm_dropout(x)
        x, _ = self.lstm2(x)
#        print("======== lstm2 =========", x.shape)
        x = self.lstm_dropout(x)
#        print("===== lstm_dropout =====", x.shape)
        x = self.logits(x)
#        print("======== logits =========", x.shape)
        return x

##
# model = CaptchaModel(vocab_size=10, dropout=0.2)
# x = torch.randn(1, 1, 50, 200)
# model.forward(x)
##


##


##

