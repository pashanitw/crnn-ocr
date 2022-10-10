##
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
##
from model import CaptchaModel
from data import build_datapipes

##
batch_size = 8
def train(device="cpu"):
    model = CaptchaModel(vocab_size=10, dropout=0.2)
    model.train()
    train_iter = build_datapipes('./captcha_images_v2')
    train_loader = DataLoader(train_iter, batch_size=8, shuffle=True)
    for batch_idx, (data, target) in enumerate(train_loader):
       # data = data.squeeze(1)
        print("******* data shape ******", data.shape, target.shape)
        data, target = data.to(device), target.to(device)
        #optimizer.zero_grad()
        output = model(data)
        print("========= out shape =========")
        print(output.shape)
        break


##
train()

##


##

