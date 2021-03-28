# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] id="hddpB7Iu8ynD"
# # Implementation of Resnet
# > In this notebook  I have implemented ResNet from scratch using Pytorch

# + id="pYlM3FI-8mZK"
#importing libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


# + id="XwerA9Bw1BZb"
# an essential block of layers which forms resnets
class ResBlock(nn.Module):
  #in_channels -> input channels,int_channels->intermediate channels
  def __init__(self,in_channels,int_channels,identity_downsample=None,stride=1):
    super(ResBlock,self).__init__()
    self.expansion = 4
    self.conv1 = nn.Conv2d(in_channels,int_channels,kernel_size=1,stride=1,padding=0)
    self.bn1 = nn.BatchNorm2d(int_channels)
    self.conv2 = nn.Conv2d(int_channels,int_channels,kernel_size=3,stride=stride,padding=1)
    self.bn2 = nn.BatchNorm2d(int_channels)
    self.conv3 = nn.Conv2d(int_channels,int_channels*self.expansion,kernel_size=1,stride=1,padding=0)
    self.bn3 = nn.BatchNorm2d(int_channels*self.expansion)
    self.relu = nn.ReLU()
    self.identity_downsample =  identity_downsample
    self.stride = stride

  def forward(self,x):
    identity = x.clone()
    x =  self.conv1(x)
    x =  self.bn1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)
    x = self.conv3(x)
    x = self.bn3(x)
    #the so called skip connections
    if self.identity_downsample is not None:
      identity = self.identity_downsample(identity)

    x += identity
    x = self.relu(x)
    return x

class ResNet(nn.Module):
  def __init__(self,block,layers,image_channels,num_classes):
    super(ResNet,self).__init__()
    self.in_channels = 64
    self.conv1 = nn.Conv2d(image_channels,64,kernel_size=7,stride=2,padding=3)
    self.bn1 =  nn.BatchNorm2d(64)
    self.relu = nn.ReLU()
    self.maxpool =  nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    #the resnet layers
    self.layer1 = self._make_layer(block,layers[0],int_channels=64,stride=1)
    self.layer2 = self._make_layer(block,layers[1],int_channels=128,stride=2)
    self.layer3 = self._make_layer(block,layers[2],int_channels=256,stride=2)
    self.layer4 = self._make_layer(block,layers[3],int_channels=512,stride=2)
    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.fc1 = nn.Linear(512*4,num_classes)

  def forward(self,x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.reshape(x.shape[0],-1)
    x = self.fc1(x)
    return x

  def _make_layer(self,block,num_res_blocks,int_channels,stride):
    identity_downsample =  None
    layers = []

    if stride!=1 or self.in_channels != int_channels*4:
      identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels,int_channels*4,
                                                    kernel_size=1,stride=stride),
                                          nn.BatchNorm2d(int_channels*4))
      layers.append(ResBlock(self.in_channels,int_channels,identity_downsample,stride))
      #this expansion size will always be 4 for all the types of ResNets
      self.in_channels =  int_channels*4

      for i in range(num_res_blocks-1):
        layers.append(ResBlock(self.in_channels,int_channels))

      return nn.Sequential(*layers)

def ResNet18(img_channel=3,num_classes=1000):
  return ResNet(ResBlock,[2,2,2,2],img_channel,num_classes)


def ResNet34(img_channel=3,num_classes=1000):
  return ResNet(ResBlock,[3,4,6,3],img_channel,num_classes)

def ResNet50(img_channel=3,num_classes=1000):
  return ResNet(ResBlock,[3,4,6,3],img_channel,num_classes)


def ResNet101(img_channel=3,num_classes=1000):
  return ResNet(ResBlock,[3,4,23,3],img_channel,num_classes)
      

def ResNet152(img_channel=3,num_classes=1000):
  return ResNet(ResBlock,[3,8,36,3],img_channel,num_classes)
  


# + colab={"base_uri": "https://localhost:8080/"} id="KAyfJzPP0lRS" outputId="2b5c3f62-6c73-4681-8182-42ae7996c847"
def test():
    net = ResNet101(img_channel=3,num_classes=1000)
    x = torch.randn(4,3,224,224)
    y = net(x).to("cuda")
    print(y.size())
test()

# + id="TNlDFB7xdQ-G"

