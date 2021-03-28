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
# # Implementation of Alexnet 
# > In this notebook  I have implemented Alexnet on CIFAR10 dataset using Pytorch on CIFAR10 dataset

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

# + colab={"base_uri": "https://localhost:8080/", "height": 100, "referenced_widgets": ["01a6bb0f11ba44f0b92d10c2de8cd05a", "a717d256db8b4b429ca13f15147f6e01", "76fdf54726754e4f90a177c34419daf8", "9beb3c568c9748d39bed9310c2e68d60", "c6ef725998a24fbfbb613b42ac65047d", "d31c50ce008a44d5ba997610e78f0659", "d4cb7333804c4e3eb5e300d71bbf1a4d", "2e07bddcc84e4973a7a6ceac2cf5d7c2"]} id="ssSniLreQvn3" outputId="070dc6e6-da6b-4c32-f927-c5da994602d5"
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

# + id="1OYG5EatRRIG"
class AlexNet(ImageClassificationBase):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10),
        )
 
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# -

# A custom function to pick a default device

# + id="03gk0YLET2Sf"
def get_default_device():
  """Pick GPU if available else CPU"""
  if torch.cuda.is_available():
    return torch.device('cuda')
  else:
    return torch.device('cpu')


# + colab={"base_uri": "https://localhost:8080/"} id="n8-rq1DSYXZ_" outputId="aa4d690e-b856-4b64-bbe1-b045f76300fe"
device = get_default_device()
device


# + id="HPXag6O6YmnC"
def to_device(data,device):
  """Move tensors to chosen device"""
  if isinstance(data,(list,tuple)):
    return [to_device(x,device) for x in data]
  return data.to(device,non_blocking=True)


# + colab={"base_uri": "https://localhost:8080/"} id="k1zzFCWKYaQN" outputId="e64f1885-262b-4f72-c198-9261e82d61bb"
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


# + colab={"base_uri": "https://localhost:8080/"} id="gEmw6_yfTMnB" outputId="da3ba6d3-c876-4895-e2f9-b3b849af29db"
train_loader = DeviceDataLoader(train_loader,device)
val_loader = DeviceDataLoader(val_loader,device)
model = AlexNet()
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


# + colab={"base_uri": "https://localhost:8080/"} id="w6EPqepdbVC7" outputId="90f23377-80d4-4b5f-ed5f-2985b9e33b41"
history = [evaluate(model, val_loader)]
history

# + colab={"base_uri": "https://localhost:8080/"} id="U_iPH-HBbXsz" outputId="2ab00313-9078-4a99-cd53-5a194a034f4d"
history = fit(3,000.1,model,train_loader,val_loader)

# + id="TNlDFB7xdQ-G"

