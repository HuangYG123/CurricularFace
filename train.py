import builtins
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from config import configurations
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from head.metrics import ArcFace, CurricularFace
from util.utils import separate_irse_bn_paras, separate_resnet_bn_paras, get_time, AverageMeter, accuracy
from dataset.datasets import FaceDataset
from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
import sys
import time
import numpy as np
import scipy
import pickle

def adjust_learning_rate(optimizer, epoch, cfg):
    """Decay the learning rate based on schedule"""
    lr = cfg['LR']
    for milestone in cfg['STAGES']:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    cfg = configurations[1]
    ngpus_per_node = torch.cuda.device_count()
    world_size = cfg['WORLD_SIZE']
    cfg['WORLD_SIZE'] = ngpus_per_node * world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg))
    
def main_worker(gpu, ngpus_per_node, cfg):
    SEED = cfg['SEED'] # random seed for reproduce results
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    cfg['GPU'] = gpu
    if gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    cfg['RANK'] = cfg['RANK'] * ngpus_per_node + gpu
    dist.init_process_group(backend=cfg['DIST_BACKEND'], init_method = cfg["DIST_URL"], world_size=cfg['WORLD_SIZE'], rank=cfg['RANK'])
    
    # Data loading code
    batch_size = int(cfg['BATCH_SIZE'] / ngpus_per_node)
    workers = int((cfg['NUM_WORKERS'] + ngpus_per_node - 1) / ngpus_per_node)
    DATA_ROOT = cfg['DATA_ROOT'] # the parent root where your train/val/test data are stored
    RECORD_DIR = cfg['RECORD_DIR']
    RGB_MEAN = cfg['RGB_MEAN'] # for normalize inputs
    RGB_STD = cfg['RGB_STD']
    train_transform = transforms.Compose([
           transforms.RandomHorizontalFlip(),
           transforms.ToTensor(),
           transforms.Normalize(mean = RGB_MEAN,
                            std = RGB_STD),])
    dataset_train = FaceDataset(DATA_ROOT, RECORD_DIR, train_transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle = (train_sampler is None), num_workers=workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    SAMPLE_NUMS = dataset_train.get_sample_num_of_each_class()
    NUM_CLASS = len(train_loader.dataset.classes)
    print("Number of Training Classes: {}".format(NUM_CLASS))
 
    #======= model & loss & optimizer =======#
    BACKBONE_DICT = {'ResNet_50': ResNet_50, 
                     'ResNet_101': ResNet_101, 
                     'ResNet_152': ResNet_152,
                     'IR_50': IR_50, 
                     'IR_101': IR_101, 
                     'IR_152': IR_152,
                     'IR_SE_50': IR_SE_50, 
                     'IR_SE_101': IR_SE_101, 
                     'IR_SE_152': IR_SE_152}
    BACKBONE_NAME = cfg['BACKBONE_NAME']
    INPUT_SIZE = cfg['INPUT_SIZE']
    assert INPUT_SIZE == [112, 112]
    backbone = BACKBONE_DICT[BACKBONE_NAME](INPUT_SIZE)
    print("=" * 60)
    print(backbone)
    print("{} Backbone Generated".format(BACKBONE_NAME))
    print("=" * 60)
    HEAD_DICT = {'ArcFace': ArcFace,
                 'CurricularFace': CurricularFace}
    HEAD_NAME = cfg['HEAD_NAME']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE'] # feature dimension
    head = HEAD_DICT[HEAD_NAME](in_features = EMBEDDING_SIZE, out_features = NUM_CLASS)
    print("=" * 60)
    print(head)
    print("{} Head Generated".format(HEAD_NAME))
    print("=" * 60)

   #--------------------optimizer-----------------------------
    if BACKBONE_NAME.find("IR") >= 0:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(backbone) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
    else:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(backbone) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability

    LR = cfg['LR'] # initial LR
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    MOMENTUM = cfg['MOMENTUM']
    optimizer = optim.SGD([{'params': backbone_paras_wo_bn + list(head.parameters()), 'weight_decay': WEIGHT_DECAY}, {'params': backbone_paras_only_bn}], lr = LR, momentum = MOMENTUM)
    print("=" * 60)
    print(optimizer)
    print("Optimizer Generated")
    print("=" * 60)
  
   
    # loss
    LOSS_NAME = cfg['LOSS_NAME']
    LOSS_DICT = {'Softmax': nn.CrossEntropyLoss()}
    loss = LOSS_DICT[LOSS_NAME].cuda(gpu)
    print("=" * 60)
    print(loss)
    print("{} Loss Generated".format(loss))
    print("=" * 60)
    
    torch.cuda.set_device(cfg['GPU'])
    backbone.cuda(cfg['GPU'])
    head.cuda(cfg['GPU'])

    #optionally resume from a checkpoint 
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT'] # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint
    if BACKBONE_RESUME_ROOT:
        print("=" * 60)
        if os.path.isfile(BACKBONE_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            loc = 'cuda:{}'.format(cfg['GPU'])
            backbone.load_state_dict(torch.load(BACKBONE_RESUME_ROOT, map_location=loc))
            if os.path.isfile(HEAD_RESUME_ROOT):
                print("Loading Head Checkpoint '{}'".format(HEAD_RESUME_ROOT))
                checkpoint = torch.load(HEAD_RESUME_ROOT, map_location=loc)
                cfg['START_EPOCH'] = checkpoint['EPOCH']
                head.load_state_dict(checkpoint['HEAD'])
                optimizer.load_state_dict(checkpoint['OPTIMIZER'])
        else:
            print("No Checkpoint Found at '{}' and '{}'. Please Have a Check or Continue to Train from Scratch".format(BACKBONE_RESUME_ROOT, HEAD_RESUME_ROOT))
        print("=" * 60)


    backbone = torch.nn.parallel.DistributedDataParallel(backbone, device_ids=[cfg['GPU']])
    head = torch.nn.parallel.DistributedDataParallel(head, device_ids=[cfg['GPU']])

     # checkpoint and tensorboard dir
    MODEL_ROOT = cfg['MODEL_ROOT'] # the root to buffer your checkpoints
    LOG_ROOT = cfg['LOG_ROOT'] # the root to log your train/val status
    STAGES = cfg['STAGES'] # epoch stages to decay learning rate
    if not os.path.exists(MODEL_ROOT):
        os.makedirs(MODEL_ROOT)
    if not os.path.exists(LOG_ROOT):
        os.makedirs(LOG_ROOT)
    writer = SummaryWriter(LOG_ROOT) # writer for buffering intermedium results
    # train
    for epoch in range(cfg['START_EPOCH'], cfg['NUM_EPOCH']):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, cfg)

        #train for one epoch
        train(train_loader, backbone, head, loss, optimizer, epoch, cfg, writer)
        print("=" * 60)
        print("Save Checkpoint...")
        if cfg['RANK'] % ngpus_per_node == 0:
            torch.save(backbone.module.state_dict(), os.path.join(MODEL_ROOT, "Backbone_{}_Epoch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, get_time())))
            save_dict = {'EPOCH': epoch+1,
                         'HEAD': head.module.state_dict(),
                         'OPTIMIZER': optimizer.state_dict()}
            torch.save(save_dict, os.path.join(MODEL_ROOT, "Head_{}_Epoch_{}_Time_{}_checkpoint.pth".format(HEAD_NAME, epoch + 1, get_time())))
 
def train(train_loader, backbone, head, criterion, optimizer, epoch, cfg, writer):
    DISP_FREQ = 100  # 100 batch
    batch = 0  # batch index
    backbone.train()  # set to training mode
    head.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for inputs, labels in tqdm(iter(train_loader)):
        # compute output
        start_time=time.time()
        inputs = inputs.cuda(cfg['GPU'], non_blocking=True)
        labels = labels.cuda(cfg['GPU'], non_blocking=True)
        features, conv_features = backbone(inputs)

        outputs, original_logits = head(features, labels)
        loss = criterion(outputs, labels)
        end_time = time.time()
        duration = end_time - start_time
        if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
            print("batch inference time", duration)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(original_logits.data, labels, topk = (1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.data.item(), inputs.size(0))
        top5.update(prec5.data.item(), inputs.size(0))
            # dispaly training loss & acc every DISP_FREQ
        if ((batch + 1) % DISP_FREQ == 0) or batch == 0:
            print("=" * 60)
            print('Epoch {}/{} Batch {}/{}\t'
                            'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                epoch + 1, cfg['NUM_EPOCH'], batch + 1, len(train_loader), loss = losses, top1 = top1, top5 = top5))
            print("=" * 60)
        sys.stdout.flush()
        batch += 1 # batch index
    epoch_loss = losses.avg
    epoch_acc = top1.avg
    print("=" * 60)
    print('Epoch: {}/{}\t''Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch + 1, cfg['NUM_EPOCH'], loss = losses, top1 = top1, top5 = top5))
    sys.stdout.flush()
    print("=" * 60)
    if cfg['RANK'] == 0:
        writer.add_scalar("Training_Loss", epoch_loss, epoch + 1)
        writer.add_scalar("Training_Accuracy", epoch_acc, epoch + 1)
        writer.add_scalar("Top1", top1.avg, epoch+1)
        writer.add_scalar("Top5", top5.avg, epoch+1)

if __name__ == '__main__':
    main()
