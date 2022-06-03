import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import transforms
from torchvision.transforms import ToTensor, RandomCrop
import torchvision.models as models
model = models.efficientnet_v2_m(pretrained=True)    #efficientnet_b3
model.avgpool=nn.AdaptiveAvgPool2d((1,1))
for i,layer in enumerate(list(model.classifier)):
    lastlayer=layer
feature = lastlayer.in_features
model.classifier = nn.Sequential(
    nn.BatchNorm1d(feature),
    nn.Linear(feature, 1000),
)
