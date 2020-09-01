'''
created_by: Glenn Kroegel
date: 23 August 

https://stackoverflow.com/questions/49995594/half-precision-floating-point-arithmetic-on-intel-chips
https://www.kaggle.com/cswwp347724/wavenet-pytorch

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from loading import *
import numpy as np
from tqdm import tqdm
import math
import json
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
num_classes = 7
quantization_channels = 32
torch.autograd.set_detect_anomaly(True)

mode = 'fp32'
if mode == 'fp16':
    from apex import amp

class ConvBlock(nn.Module):
    '''gated, filter and joining conv'''
    def __init__(self, channels, kernel_sz, dilation, bias=False):
        super(ConvBlock, self).__init__()
        padding = int((dilation*(kernel_sz-1))/2)
        self.filter_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_sz,
            dilation=dilation,
            padding=padding,
            bias=bias
        )
        self.gated_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_sz,
            dilation=dilation,
            padding=padding,
            bias=bias
        )
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            dilation=1,
            bias = bias)

    def forward(self, x):
        out = torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gated_conv(x))
        out = self.conv(out)
        return out

class ResBlocks(nn.Module):
    def __init__(self, channels, depth):
        super(ResBlocks, self).__init__()
        conv_blocks = []
        for i in range(depth):
            d = 2 ** i
            conv_blocks.append(ConvBlock(channels=channels, kernel_sz=3, dilation=d))
        self.conv_blocks = nn.ModuleList(conv_blocks)

    def forward(self, x):
        skip = []
        res = x
        for block in self.conv_blocks:
            x = block(x)
            res = res + x
            res = F.group_norm(res, 8)
            skip.append(res)
        skip = torch.stack(skip).sum(0)
        skip = F.group_norm(skip, 8)
        return skip

class WaveNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth):
        super(WaveNetBlock, self).__init__()
        self.conv_in = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        self.res_blocks = ResBlocks(out_channels, depth)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res_blocks(x)
        return x

class WaveNet(nn.Module):
    def __init__(self, channels, depth):
        super(WaveNet, self).__init__()
        blocks = [WaveNetBlock(channels[i], channels[i+1], depth) for i in range(len(channels)-1)]
        self.blocks = nn.Sequential(*blocks)
        c = channels[-1]
        self.conv1 = nn.Conv1d(
            in_channels=c,
            out_channels=c,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=c,
            out_channels=c,
            kernel_size=3,
            padding=1
        )
        self.act = nn.GELU()
        self.final_layers = nn.Sequential(self.conv1, self.act, self.conv2)


    def forward(self, x):
        x = self.blocks(x)
        x = self.act(x)
        x = self.final_layers(x)
        return x

class QuantizationHead(nn.Module):
    def __init__(self, in_c, quantization_channels):
        super(QuantizationHead, self).__init__()
        self.l_out = nn.Linear(in_c, quantization_channels)

    def forward(self, x):
        '''in_shp: (bs, c, sl) out_shp: (bs, sl, q_c)'''
        x = x.permute(0, 2, 1)
        x = self.l_out(x)
        return x

class ClassificationHead(nn.Module):
    def __init__(self, in_c, num_classes):
        super(ClassificationHead, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_c, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        x = torch.sigmoid(self.conv(x))
        return x

class LogitsHead(nn.Module):
    def __init__(self, in_c, num_classes):
        super(LogitsHead, self).__init__()
        self.l_out = nn.Linear(in_c, num_classes)

    def forward(self, x):
        return x

class Model(nn.Module):
    def __init__(self, channels, depth, num_classes):
        super(Model, self).__init__()
        self.encoder = WaveNet(channels, depth)
        self.head = ClassificationHead(in_c=channels[-1], num_classes=num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        x = x.permute(0, 2, 1)
        return x

def loss_function(output, y):
    loss = F.binary_cross_entropy(output, y)
    return loss

def evaluate(loader, model):
    props = {'loss': 0}
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i % 5 == 0:
                print(i)
            x, y = batch
            x = x.to(device)
            # y = y.to(device)
            y = F.one_hot(y, num_classes=num_classes).float().to(device)
            outp = model(x)
            loss = loss_function(outp, y)
            props['loss'] += loss.item()
    L = len(loader)
    props = {k:v/L for k,v in props.items()}
    return props

def train(loader, model, optimizer, scheduler, mode):
    '''training loop'''
    props = {'loss': 0, 'lr': 0}
    model.train()
    for i, batch in enumerate(loader):
        optimizer.zero_grad()
        if i % 1 == 0:
            print(i)
        x, y = batch
        x = x.to(device)
        # y = y.to(device)
        y = F.one_hot(y, num_classes=num_classes).float().to(device)
        outp = model(x)
        loss = loss_function(outp, y)
        if mode == 'fp16':
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()
        props['loss'] += loss.item()
    L = len(loader)
    props = {k:v/L for k,v in props.items()}
    props['lr'] = round(scheduler.get_last_lr()[0], 3)
    return props

def status(epoch, train_props, cv_props, epochs):
    '''generate summary during training'''
    s0 = 'epoch {0}/{1}\n'.format(epoch, epochs)
    s1, s2 = '',''
    for k,v in train_props.items():
        s1 = s1 + 'train_'+ k + ': ' + str(v) + ' '
    for k,v in cv_props.items():
        s2 = s2 + 'cv_'+ k + ': ' + str(v) + ' '
    print(s0 + s1 + s2)

def main():
    train_iter = torch.load('train_loader.pt')
    test_iter = torch.load('cv_loader.pt')
    model = Model(channels=[2, 16], depth=4, num_classes=num_classes).to(device)
    print(model)
    print(len(train_iter), len(test_iter))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-4)
    epochs = 30
    best_loss = np.inf

    if mode == 'fp16':
        opt_level = 'O1'
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level, loss_scale='dynamic')

    restore = False
    if restore:
        checkpoint = torch.load('encoder.pth.tar')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        amp.load_state_dict(checkpoint['amp'])
        best_loss = checkpoint['loss']

    # Training loop
    for epoch in tqdm(range(epochs)):
        train_props = train(train_iter, model, optimizer, scheduler, mode)
        cv_props = evaluate(test_iter, model)
        status(epoch, train_props, cv_props, epochs)

        loss = train_props['loss']
        if loss < best_loss:
            best_loss = loss
            amp_ = amp.state_dict() if mode == 'fp16' else None
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp_,
                'loss': loss
            }
            print('checkpointing..')
            torch.save(checkpoint, 'encoder.pth.tar')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('cancelling..')