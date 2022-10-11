##
import torch
import torch.nn as nn
##
class Ctc_Loss(nn.Module):
    def __init__(self, vocab_size, blank=0, reduction="mean"):
        super(Ctc_Loss, self).__init__()
        self.vocab_size = vocab_size
        self.blank = blank
        self.reduction = reduction
    def forward(self, y_pred, y_target):
        y_pred = y_pred.permute(1, 0, 2)
        y_pred = y_pred.log_softmax(2)
        T = y_pred.shape[0]
        N = y_pred.shape[1]
        S = y_target.shape[1]
        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
        target_lengths = torch.full(size=(N,), fill_value=S, dtype=torch.long)
        loss = nn.functional.ctc_loss(y_pred, y_target, input_lengths, target_lengths,blank=self.blank, reduction=self.reduction)
        return loss
##
# logits = torch.randn(1, 10) #ypred
# gt = torch.tensor([[1, 2, 3, 4, 5]]) #ytrue
# input_lengths = 10
# target_lengths = 5
# print(logits.shape, gt.shape, input_lengths, target_lengths)
# ##
# # Target are to be padded
# T = 50      # Input sequence length
# C = 20      # Number of classes (including blank)
# N = 16      # Batch size
# S = 30      # Target sequence length of longest target in batch (padding length)
# S_min = 10  # Minimum target length, for demonstration purposes
# # T is no of time steps
# #for each time stepp we have 20 classes for each input in the batch
# input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
# target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)
# input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
# print(input.shape,target.shape)
# target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
# print("======= lengths =======")
# print(input.shape)
# print(target.shape)
# print(input_lengths)
# print(target_lengths)
##


##

