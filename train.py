##
from model import CaptchaModel
from preprocess import create_dataloaders, getVocabSize
from ctc_loss import Ctc_Loss
import torch.optim as optim
##
batch_size = 16
vocab_len = getVocabSize()

def train(epochs,device="cpu"):
    train_loader, val_loader=create_dataloaders()
    model = CaptchaModel(vocab_size=vocab_len, dropout=0.2)
    model = model.to(device)
    criterion = Ctc_Loss(vocab_size=vocab_len)
    optimizer = optim.Adam(model.parameters())
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            print(loss)
            running_loss += loss.item()
            if batch_idx % 10 == 9:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, batch_idx + 1, running_loss / 10))
                running_loss = 0.0
    print('Finished Training')

##
train(epochs=5)

##


##


##

