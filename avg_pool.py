##
import torch

import torch.nn as nn

item= torch.randn(10,32,3,3)

pool = nn.AvgPool2d(3)
# note: the kernel size equals the feature map dimensions in the previous layer

output = pool(item)

output = output.squeeze()

print(output.size())
##

