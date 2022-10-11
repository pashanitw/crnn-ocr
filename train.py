##
from model import CaptchaModel
from preprocess import create_dataloaders, getVocabSize
from ctc_loss import Ctc_Loss
import torch.optim as optim
##
batch_size = 16
vocab_len = getVocabSize()

def train(epochs,train_loader=None, val_loader=None, device="cpu" ):
    model = CaptchaModel(vocab_size=vocab_len, dropout=0.2)
    model = model.to(device)
    criterion = Ctc_Loss(vocab_size=vocab_len)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        train_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            print(loss)
            train_loss += loss.item()
            running_loss += loss.item()
            if batch_idx % 10 == 9:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, batch_idx + 1, running_loss / 10))
                running_loss = 0.0
        valid_loss = 0.0
        model.eval()
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            out = model(images)
            loss = criterion(out, labels)
            valid_loss += loss.item()
        print(f'Epoch {epoch + 1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(val_loader)}')

    print('Finished Training')

##
train_loader, val_loader=create_dataloaders()
train(epochs=10,  train_loader=train_loader, val_loader=val_loader, device="cpu")
##


##


##

