import sys
import os
import warnings
from model import AMSA_CAFFNet
from utils import save_checkpoint
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import argparse
import json
import cv2
import dataset
import time

os.environ['CUDA_VISIBLE_DEVICES']="1"

parser = argparse.ArgumentParser(description='PyTorch AMSA-CAFF Net')

parser.add_argument('--train_json',default='jsons/train.json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('--val_json',default='jsons/val.json', metavar='VAL',
                    help='path to val json')

def main():

    global args,best_mae,best_rmse

    best_mae = 1e6
    best_rmse = 1e6

    args = parser.parse_args()
    args.lr = 1e-5
    args.batch_size    = 1
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 200
    args.workers = 4
    args.seed = int(time.time())
    args.print_freq = 4
    with open(args.train_json, 'r') as outfile:
        train_list = json.load(outfile)
    with open(args.val_json, 'r') as outfile:
        val_list = json.load(outfile)

    torch.cuda.manual_seed(args.seed)

    model = AMSA_CAFFNet()
    model = model.cuda()
    criterion = nn.MSELoss(size_average=False).cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr,weight_decay=args.decay)

    for epoch in range(args.start_epoch, args.epochs):
        train(train_list, model, criterion, optimizer, epoch)
        prec1,prec2 = validate(val_list, model, criterion)

        is_best = prec1 < best_mae
        best_mae = min(prec1, best_mae)
        best_rmse = min(prec2, best_rmse)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_mae))
        print(' * best RMSE {mse:.3f} '
              .format(mse=best_rmse))
        save_checkpoint({
            'state_dict': model.state_dict(),
        }, is_best)

def train(train_list, model, criterion, optimizer, epoch):

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),
                       train=True,
                       seen=model.seen,
                       batch_size=args.batch_size,
                       num_workers=args.workers),
        batch_size=args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))

    model.train()
    end = time.time()

    for i,(img, target)in enumerate(train_loader):
        data_time.update(time.time() - end)

        img = img.cuda()
        output = model(img)
        target=target.float().cuda()
        loss = criterion(output, target)
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))


def validate(val_list, model, criterion):
    print ('begin val')
    val_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=1)

    model.eval()
    mae = 0
    rmse=0

    for i,(img, target) in enumerate(val_loader):
        h,w = img.shape[2:4]
        h_d = h//2
        w_d = w//2
        img_1 = img[:, :, :h_d, :w_d].cuda()
        img_2 = img[:, :, :h_d, w_d:].cuda()
        img_3 = img[:, :, h_d:, :w_d].cuda()
        img_4 = img[:, :, h_d:, w_d:].cuda()
        density_1 = model(img_1).data.cpu().numpy()
        density_2 = model(img_2).data.cpu().numpy()
        density_3 = model(img_3).data.cpu().numpy()
        density_4 = model(img_4).data.cpu().numpy()

        pred_sum = density_1.sum()+density_2.sum()+density_3.sum()+density_4.sum()

        mae += abs(pred_sum - target.sum())
        rmse += pow(pred_sum - target.sum(), 2)

    mae = mae / len(val_loader)
    rmse = pow(rmse / len(val_loader), 1 / 2)
    print(' * MAE {mae:.3f} '
          .format(mae=mae))
    print(' * MSE {mse:.3f} '
          .format(mse=rmse))

    return mae, rmse

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

if __name__ == '__main__':
    main()
