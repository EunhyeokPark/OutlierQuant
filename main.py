import os, time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

                           
import argparse
parser = argparse.ArgumentParser(description = 'ImageNet OLNet Training')                           
parser.add_argument('--abit', default = 0, type=int, metavar='N', help='number of bits for forward opeartion')

parser.add_argument('--wbit', default = 0, type=int, metavar='N', help='number of bits for backward opeartion')

parser.add_argument('--ratio', default = 0.01, type=float, metavar='N', help='outlier ratio')

parser.add_argument('--uiter', default = 0, type=int, metavar='N', help='backward update per iteration')
args = parser.parse_args()

transform_val = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std = [ 0.229, 0.224, 0.225 ]),
])

base_dir = "/SSD/ILSVRC2012"

val = datasets.ImageFolder(os.path.join(base_dir, "val_pt"), transform_val)
val_loader = torch.utils.data.DataLoader(
  val, batch_size = 400, shuffle = True, num_workers = 8, pin_memory = True)
  

import ol_alexnet
model = ol_alexnet.ol_alexnet()
model.load_state_dict(torch.load("alexnet-owt-4df8aa71.pth"))

for name, module in model.named_modules():
  if isinstance(module, ):
    module.


use_cuda = torch.cuda.is_available()
if use_cuda:
  model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count())).cuda()
  cudnn.benchmark = True

criterion = nn.CrossEntropyLoss().cuda()

from train_test import validate
from ol_module import OL_Active as OL_Active

def update():  
  for name, module in model.module.named_ol_layers():
    if isinstance(module, OL_Active):
      if args.abit != 0:
        module.ol_update(args.abit, 1 - args.ratio)
    else:
      if args.wbit != 0:
        module.ol_update(args.wbit, 1 - args.ratio)    
    
prec1 = validate(val_loader, model, criterion)
