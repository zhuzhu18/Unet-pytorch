import cv2
import torch
import numpy as np
from unet import UNet
from utils import get_ids, split_train_val, get_imgs_and_masks, batch, mkdir_p
from dice_loss import dice_coeff
import optparse
import sys
import os
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR


def train(net, iddataset, dir_img, dir_mask,
          optimizer, criterion, args):
    n_train = len(iddataset['train'])

    # criterion = nn.BCELoss()
    net.train()
    if args.gpu:
        net.to(args.device)

    train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask)

    epoch_loss = 0.
    for i, b in enumerate(batch(train, args.batchsize)):
        imgs = np.array([data[0] for data in b]).astype(np.float32)
        masks = np.array([data[1] for data in b]).astype(np.float32)

        imgs = torch.from_numpy(imgs).unsqueeze(1).float()
        masks = torch.from_numpy(masks).float()

        if args.gpu:
            imgs = imgs.to(args.device)
            masks = masks.to(args.device)
        optimizer.zero_grad()
        mask_pred = net(imgs)
        mask_prob_flat = mask_pred.view(-1)
        masks_flat = masks.view(-1)

        loss = criterion(mask_prob_flat, masks_flat)
        epoch_loss += loss.cpu().item()

        loss.backward()
        optimizer.step()
        print('training progress:{0:.4f} --- loss: {1:.6f}'.format(i*args.batchsize/n_train, loss.item()))


def eval_net(net, dataset, dir_img, dir_mask, args):
    net.eval()
    if args.gpu:
        net.to(args.device)

    total = 0
    val = get_imgs_and_masks(dataset['val'], dir_img, dir_mask)
    for i, b in enumerate(val):
        img = np.array(b[0]).astype(np.float32)
        mask = np.array(b[1]).astype(np.float32)

        img = torch.from_numpy(img)[None, None, :, :]
        mask = torch.from_numpy(mask).unsqueeze(0)

        if args.gpu:
            img = img.to(args.device)
            mask = mask.to(args.device)
        mask_pred = net(img)
        mask_pred = (mask_pred > 0.5).float()         # 得到预测的分割图

        total += dice_coeff(mask_pred, mask, args.device).cpu().item()
    current_score = total / (i+1)
    global best_score
    print('current score is %f'%current_score)
    print('best score is %f'%best_score)
    if current_score > best_score:
        best_score = current_score
        print('current best score is {}'.format(best_score))
        if args.save_cp:
            print('saving checkpoint')
            mkdir_p('checkpoint')
            torch.save(net.state_dict(), './checkpoint/unet.pth')

    return best_score

def get_args():
    parser = optparse.OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=100, type='int', help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=4, type='int', help='minibatch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=1e-3, type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load', default=False, help='load file mode')
    parser.add_option('-n', '--numclasses', dest='num_classes', default=2, type='int', help='classes to seg')
    parser.add_option('-d', '--device', dest='device', default='cuda:0', type='str', help='cuda device and number')
    parser.add_option('-v', '--val-percent', dest='val_percent', default=0.15, type='float', help='val percentage')
    parser.add_option('-s', '--save-cp', dest='save_cp', action='store_true', default=True, help='whether to save checkpoint')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()
    # dir_img = '/home/zzh/数据/mid project/raw_data'
    # dir_mask = '/home/zzh/数据/mid project/groundtruth'

    dir_img = '/home/zhuzhu/Desktop/mid project/raw_data'
    dir_mask = '/home/zhuzhu/Desktop/mid project/groundtruth'
    ids = get_ids(dir_img)          # 1,2,3,...的生成器

    iddataset = split_train_val(ids, args.val_percent)     # {'train':[23,98,59,...],'val':[12,37,48,...]}

    net = UNet(n_channels=1, n_classes=args.num_classes)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.99, weight_decay=5e-3)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.BCEWithLogitsLoss()
    if args.load:
        print('load model from checkpoint')
        net.load_state_dict(torch.load('checkpoint/unet.pth'))

    if args.gpu:
        net.to(args.device)

    best_score = 0
    for epoch in range(args.epochs):
        print('start training ==>')
        scheduler.step()
        train(net, iddataset, dir_img, dir_mask, optimizer,
              criterion, args)
        print('start testing ==>')
        eval_net(net, iddataset, dir_img, dir_mask, args)

    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)