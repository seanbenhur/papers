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
# # Implementation of VGG16
# > In this notebook  I have implemented VGG16 on CIFAR10 dataset using  Pytorch

# + id="pYlM3FI-8mZK"
#importing libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
import tqdm
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
# -

# Load the data and do standard preprocessing steps,such as resizing and converting the images into tensor

# + colab={"base_uri": "https://localhost:8080/"} id="ssSniLreQvn3" outputId="9b0473ed-4a9b-40c3-93b2-24b4191b5f40"
transform = transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485,0.456,0.406],
                                                     std=[0.229,0.224,0.225])])

train_ds = CIFAR10(root='data/',train = True,download=True,transform = transform)
val_ds = CIFAR10(root='data/',train = False,download=True,transform = transform)

batch_size = 128
train_loader = DataLoader(train_ds,batch_size,shuffle=True,num_workers=4,pin_memory=True)
val_loader = DataLoader(val_ds,batch_size,num_workers=4,pin_memory=True)


# -

# A custom utility class to print out the accuracy and losses during training and testing

# + id="7yVd2dEgKRZt"
def accuracy(outputs,labels):
  _,preds = torch.max(outputs,dim=1)
  return torch.tensor(torch.sum(preds==labels).item()/len(preds))
 
class ImageClassificationBase(nn.Module):
  def training_step(self,batch):
    images, labels = batch
    out = self(images)
    loss = F.cross_entropy(out,labels)
    return loss
  
  def validation_step(self,batch):
    images, labels = batch
    out = self(images)
    loss = F.cross_entropy(out,labels)
    acc = accuracy(out,labels)
    return {'val_loss': loss.detach(),'val_acc': acc}
  
  def validation_epoch_end(self,outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
  
  def epoch_end(self, epoch, result):
    print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


# -

# ### Creating a network

# + id="n7YBu4Vyi1XG"
VGG_types = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG_net(ImageClassificationBase):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types['VGG16'])
        
        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
            )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == int:
                out_channels = x
                
                layers += [nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                                     kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
                
        return nn.Sequential(*layers)


# -

# A custom function to pick a default device

# + id="03gk0YLET2Sf"
def get_default_device():
  """Pick GPU if available else CPU"""
  if torch.cuda.is_available():
    return torch.device('cuda')
  else:
    return torch.device('cpu')


# + colab={"base_uri": "https://localhost:8080/"} id="n8-rq1DSYXZ_" outputId="056b9e75-b75a-4d43-e0b2-ffc8f96e5257"
device = get_default_device()
device


# + id="HPXag6O6YmnC"
def to_device(data,device):
  """Move tensors to chosen device"""
  if isinstance(data,(list,tuple)):
    return [to_device(x,device) for x in data]
  return data.to(device,non_blocking=True)


# + colab={"base_uri": "https://localhost:8080/"} id="k1zzFCWKYaQN" outputId="443f633e-919d-4672-9c98-71f4436f3322"
for images, labels in train_loader:
  print(images.shape)
  images = to_device(images,device)
  print(images.device)
  break


# + id="VNcp-yclZJZk"
class DeviceDataLoader():
  """Wrap a DataLoader to move data to a device"""
  def __init__(self,dl,device):
    self.dl = dl
    self.device =  device
  def __iter__(self):
    """Yield a batch of data to a dataloader"""
    for b in self.dl:
      yield to_device(b, self.device)
  def __len__(self):
    """Number of batches"""
    return len(self.dl)


# + colab={"base_uri": "https://localhost:8080/"} id="gEmw6_yfTMnB" outputId="b48359bb-45f1-4957-ecfa-dd159e47c10f"
train_loader = DeviceDataLoader(train_loader,device)
val_loader = DeviceDataLoader(val_loader,device)
model = VGG_net(in_channels=3,num_classes=10)
to_device(model,device)


# -

# ### Training the model

# + id="4yfuZgkJbHt-"
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)
 
def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    train_losses =[]
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# + colab={"base_uri": "https://localhost:8080/"} id="w6EPqepdbVC7" outputId="f2689347-5b0f-4a90-ab22-ba78039d3dce"
history = [evaluate(model, val_loader)]
history

# + id="wvn22QDKWyZv"
#history = fit(2,0.1,model,train_loader,val_loader)

# + id="TNlDFB7xdQ-G"

