##
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
##
from model import CaptchaModel
from data import create_dataloaders, getVocabSize
from ctc_loss import Ctc_Loss
##
batch_size = 8
vocab_len = getVocabSize()

def train(device="cpu"):
    train_loader, val_loader=create_dataloaders()
    model = CaptchaModel(vocab_size=vocab_len, dropout=0.2)
    criterion = Ctc_Loss(vocab_size=vocab_len)
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        out = model(images)
        loss = criterion(out, labels)
        print("============= loss is ======")
        print(loss)
        print(out.shape)
        break
##
train()

##


##


##

