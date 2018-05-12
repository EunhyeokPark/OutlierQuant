import time 
from tqdm import *
from torch.autograd import Variable

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count
    
def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0 / batch_size))
  
  return res  
 
def validation(net, loader, loss_func, use_cuda, topk = (1,)):
  print("this module is deprecated...use TrainTest.py")
  net.eval()
  test_loss = 0
  batch_time = AverageMeter()
  losses = AverageMeter()
  
  topk_lst = []
  for k in topk:
    topk_lst.append(AverageMeter())

  total = 0
  end = time.time()
  for (inputs, targets) in tqdm(loader):
    if use_cuda:
      inputs, targets = inputs.cuda(), targets.cuda()
    inputs, targets = Variable(inputs, volatile=True), Variable(targets)

    # compute output
    outputs = net(inputs)
    loss = loss_func(outputs, targets)

    # measure accuracy and record loss
    topk_rtn = accuracy(outputs.data, targets.data, topk)   
    for idx, topk_val in enumerate(topk_rtn):
      topk_lst[idx].update(topk_val[0], inputs.size(0))
    losses.update(loss.data[0], inputs.size(0))    

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

  # Save checkpoint.
  print("validation finished")
  return (batch_time.avg, losses.avg) + tuple(topk_val.avg for topk_val in topk_lst)