# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn.functional as F
import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

from method import Img_D, Ins_D, Img_D_res, Ins_D_res
import itertools

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='vgg16', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')                      
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and diaplay
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')
# cityscape
  parser.add_argument('--alpha', dest='alpha',
                      help='teacher model update weight',
                      default=0.99, type=float)

  parser.add_argument('--beta', dest='beta',
                      help='beta',
                      default=0.8, type=float)

  parser.add_argument('--w_cons', dest='w_cons',
                      help='w_cons',
                      default=0.1, type=float)
  parser.add_argument('--w_img', dest='w_img',
                      help='w_img',
                      default=0.1, type=float)

  parser.add_argument('--w_ins', dest='w_ins',
                      help='w_ins',
                      default=0.1, type=float)

  parser.add_argument('--lr_GD', dest='lr_GD',
                      help='lr_GD',
                      default=0.001, type=float)

  # # KITTI
  # parser.add_argument('--alpha', dest='alpha',
  #                     help='teacher model update weight',
  #                     default=0.99, type=float)
  #
  # parser.add_argument('--beta', dest='beta',
  #                     help='beta',
  #                     default=0.9, type=float)
  #
  # parser.add_argument('--w_cons', dest='w_cons',
  #                     help='w_cons',
  #                     default=0.1, type=float)
  # parser.add_argument('--w_img', dest='w_img',
  #                     help='w_img',
  #                     default=2, type=float)
  #
  # parser.add_argument('--w_ins', dest='w_ins',
  #                     help='w_ins',
  #                     default=1, type=float)
  #
  # parser.add_argument('--lr_GD', dest='lr_GD',
  #                     help='lr_GD',
  #                     default=0.001, type=float)

  # # SIM
  # parser.add_argument('--alpha', dest='alpha',
  #                     help='teacher model update weight',
  #                     default=0.99, type=float)
  #
  # parser.add_argument('--beta', dest='beta',
  #                     help='beta',
  #                     default=0.9, type=float)
  #
  # parser.add_argument('--w_cons', dest='w_cons',
  #                     help='w_cons',
  #                     default=0.1, type=float)
  # parser.add_argument('--w_img', dest='w_img',
  #                     help='w_img',
  #                     default=0.01, type=float)
  #
  # parser.add_argument('--w_ins', dest='w_ins',
  #                     help='w_ins',
  #                     default=0.1, type=float)
  #
  # parser.add_argument('--lr_GD', dest='lr_GD',
  #                     help='lr_GD',
  #                     default=0.001, type=float)

  # # watercolor
  # parser.add_argument('--alpha', dest='alpha',
  #                     help='teacher model update weight',
  #                     default=0.99, type=float)
  #
  # parser.add_argument('--beta', dest='beta',
  #                     help='beta',
  #                     default=0.8, type=float)
  #
  # parser.add_argument('--w_cons', dest='w_cons',
  #                     help='w_cons',
  #                     default=0.1, type=float)
  # parser.add_argument('--w_img', dest='w_img',
  #                     help='w_img',
  #                     default=0.1, type=float)
  #
  # parser.add_argument('--w_ins', dest='w_ins',
  #                     help='w_ins',
  #                     default=0.01, type=float)
  #
  # parser.add_argument('--lr_GD', dest='lr_GD',
  #                     help='lr_GD',
  #                     default=0.001, type=float)

  # # clipart
  # parser.add_argument('--alpha', dest='alpha',
  #                     help='teacher model update weight',
  #                     default=0.99, type=float)
  #
  # parser.add_argument('--beta', dest='beta',
  #                     help='beta',
  #                     default=0.9, type=float)
  #
  # parser.add_argument('--w_cons', dest='w_cons',
  #                     help='w_cons',
  #                     default=0.1, type=float)
  # parser.add_argument('--w_img', dest='w_img',
  #                     help='w_img',
  #                     default=0.1, type=float)
  #
  # parser.add_argument('--w_ins', dest='w_ins',
  #                     help='w_ins',
  #                     default=0.01, type=float)
  #
  # parser.add_argument('--lr_GD', dest='lr_GD',
  #                     help='lr_GD',
  #                     default=0.001, type=float)
  #
  # # cartoon
  # parser.add_argument('--alpha', dest='alpha',
  #                     help='teacher model update weight',
  #                     default=0.99, type=float)
  #
  # parser.add_argument('--beta', dest='beta',
  #                     help='beta',
  #                     default=0.9, type=float)
  #
  # parser.add_argument('--w_cons', dest='w_cons',
  #                     help='w_cons',
  #                     default=0.01, type=float)
  # parser.add_argument('--w_img', dest='w_img',
  #                     help='w_img',
  #                     default=0.01, type=float)
  #
  # parser.add_argument('--w_ins', dest='w_ins',
  #                     help='w_ins',
  #                     default=0.01, type=float)
  #
  # parser.add_argument('--lr_GD', dest='lr_GD',
  #                     help='lr_GD',
  #                     default=0.001, type=float)

  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

  elif args.dataset == "cityscape":
      print('loading our dataset...........')
      args.s_imdb_name = "cityscape_2007_train_s"
      args.t_imdb_name = "cityscape_2007_train_t"
      args.s_imdbtest_name="cityscape_2007_test_s500"
      args.t_imdbtest_name="cityscape_2007_test_t"

      args.imdb_name = args.t_imdb_name
      args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

  elif args.dataset == "KITTI":
      print('loading our dataset...........')
      args.s_imdb_name = "KITTI_2007_trainall"
      args.t_imdb_name = "KITTI_cityscape_2007_train_s"
      args.s_imdbtest_name = "cityscape_2007_test_s500"
      args.t_imdbtest_name = "cityscape_2007_test_t"

      args.imdb_name = args.t_imdb_name
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      #args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

  elif args.dataset == "CK":
      print('loading our dataset...........')
      args.s_imdb_name = "KITTI_cityscape_2007_train_s"
      args.t_imdb_name = "KITTI_2007_trainall"
      args.s_imdbtest_name = "KITTI_cityscape_2007_test_s500"
      args.t_imdbtest_name = "KITTI_2007_train500"

      args.imdb_name = args.t_imdb_name
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      # args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

  elif args.dataset == "SIM":
      print('loading our dataset...........')
      args.s_imdb_name = "SIM_2012_train_s"
      args.t_imdb_name = "SIM_cityscape_2007_train_s"
      args.s_imdbtest_name="SIM_cityscape_2007_test_s500"
      args.t_imdbtest_name="SIM_cityscape_2007_test_t"

      args.imdb_name = args.t_imdb_name
      #args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "voc2cartoon":
      args.s_imdb_name = "voc_2007_trainval"
      args.t_imdb_name = "voc_cartoon_2007_train_cartoon"
      args.s_imdbtest_name = "voc_2007_test"
      args.t_imdbtest_name = "voc_cartoon_2007_test_cartoon"
      args.imdb_name = args.t_imdb_name
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "voc2wc":
      args.s_imdb_name = "voc_2007_trainval"
      args.t_imdb_name = "voc_wc_2007_train_wc"
      args.s_imdbtest_name = "voc_2007_test"
      args.t_imdbtest_name = "voc_wc_2007_test_wc"
      args.imdb_name = args.t_imdb_name
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "voc2comic":
      args.s_imdb_name = "voc_2007_trainval"
      args.t_imdb_name = "voc_comic_2007_train_comic"
      args.s_imdbtest_name = "voc_2007_test"
      args.t_imdbtest_name = "voc_comic_2007_test_comic"
      args.imdb_name = args.t_imdb_name
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "voc2clipart":
      args.s_imdb_name = "voc_2007_trainval"
      args.t_imdb_name = "voc_clipart_2007_traintest1k"
      args.s_imdbtest_name = "voc_2007_test"
      args.t_imdbtest_name = "voc_clipart_2007_traintest1k"
      args.imdb_name = args.t_imdb_name
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "vg":
      # train sizes: train, smalltrain, minitrain
      # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
  train_size = len(roidb)

  print('{:d} roidb entries'.format(len(roidb)))

  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  sampler_batch = sampler(train_size, args.batch_size)

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
    fasterRCNN_T = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    fasterRCNN_T = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()
  fasterRCNN_T.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr

  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.cuda:
    fasterRCNN.cuda()
    fasterRCNN_T.cuda()
      
  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  criterion_GAN = torch.nn.BCELoss()
  if args.net == 'vgg16':
      netD = Img_D().cuda()
      netD_ins = Ins_D().cuda()
  elif args.net == 'res101':
      netD = Img_D_res().cuda()
      netD_ins = Ins_D_res().cuda()
  else:
      print('wrong net')
      pdb.set_trace()
  optimizer_D = torch.optim.Adam(itertools.chain(netD.parameters(), netD_ins.parameters()),
                                 lr=args.lr_GD, betas=(0.5, 0.999))

  rois_temp = torch.zeros(1, 1).cuda()
  not_first = False
  domain_data = 0
  print('alpha, beta, w_cons, w_img, w_ins, lr_GD')
  print(args.alpha, args.beta, args.w_cons, args.w_img, args.w_ins, args.lr_GD)

  if args.resume:
    load_name = os.path.join(output_dir,
      'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    fasterRCNN.load_state_dict(checkpoint['model'])
    fasterRCNN_T.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))

  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)
    fasterRCNN_T = nn.DataParallel(fasterRCNN_T)

  iters_per_epoch = int(train_size / args.batch_size)

  if args.use_tfboard:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter("logs")

  for param_t in fasterRCNN_T.parameters():
      # pdb.set_trace()
      param_t.requires_grad = False

  for epoch in range(args.start_epoch, args.max_epochs + 1):
    # setting to train mode
    fasterRCNN.train()
    fasterRCNN_T.train()
    loss_temp = 0
    start = time.time()

    if epoch % (args.lr_decay_step + 1) == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

    data_iter = iter(dataloader)
    for step in range(iters_per_epoch):
      data = next(data_iter)
      with torch.no_grad():
              im_data.resize_(data[0].size()).copy_(data[0])
              im_info.resize_(data[1].size()).copy_(data[1])
              gt_boxes.resize_(4,1,5).zero_()
              num_boxes.resize_(4).zero_()

      if not_first and args.net == 'vgg16' and args.dataset != 'voc2cartoon':
          domain_data = 0.5*domain_data + 0.5*im_data.mean(0).expand_as(im_data)
      else:
          domain_data = im_data.mean(0).expand_as(im_data)
          not_first = True
      im_data2 = args.beta * im_data + (1 - args.beta) * domain_data



      fasterRCNN.zero_grad()

      rois1, cls_prob1, bbox_pred1, base_feat1, pooled_feat1, cls_score1 = fasterRCNN_T(im_data, im_info, gt_boxes, num_boxes, rois_temp)



      rois, cls_prob, bbox_pred, base_feat, pooled_feat, cls_score = fasterRCNN(im_data2, im_info, gt_boxes, num_boxes, rois1)
      # pdb.set_trace()


      cons_loss = -F.log_softmax(cls_score, 1).mul(F.softmax((cls_score1.detach()), 1)).sum(0).sum(0) / cls_score.size(0)

      #image DA
      target_real = Variable(torch.ones(base_feat.size(0), base_feat.size(2), base_feat.size(3)).cuda(), requires_grad=False)
      target_fake = Variable(torch.zeros(base_feat.size(0), base_feat.size(2), base_feat.size(3)).cuda(),requires_grad=False)

      pred_real = netD(base_feat1.detach())
      loss_D_real = criterion_GAN(pred_real, target_real)
      pred_fake = netD(base_feat)
      loss_D_fake = criterion_GAN(pred_fake, target_fake)
      img_loss = loss_D_real.mean()+loss_D_fake.mean()

      # instance DA
      ins_target_real = Variable(torch.ones(pooled_feat.size(0)).long().cuda(), requires_grad=False)
      ins_target_fake = Variable(torch.zeros(pooled_feat.size(0)).long().cuda(), requires_grad=False)

      ins_pred_real = netD_ins(pooled_feat1.detach())
      ins_loss_D_real = F.cross_entropy(ins_pred_real, ins_target_real)
      ins_pred_fake = netD_ins(pooled_feat)
      ins_loss_D_fake = F.cross_entropy(ins_pred_fake, ins_target_fake)
      ins_loss = ins_loss_D_real.mean() + ins_loss_D_fake.mean()

      loss = args.w_img*img_loss.mean() + args.w_ins*ins_loss.mean() + args.w_cons*cons_loss.mean()

      loss_temp += loss.item()
      # backward
      optimizer_D.zero_grad()
      optimizer.zero_grad()
      loss.backward()
      if args.net == "vgg16":
          clip_gradient(fasterRCNN, 10.)
      optimizer.step()
      optimizer_D.step()

      for param_s, param_t in zip(fasterRCNN.parameters(), fasterRCNN_T.parameters()):
          # pdb.set_trace()
          param_t.data = args.alpha*param_t.data.detach() + (1-args.alpha)*param_s.data.detach()

      if step % args.disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= (args.disp_interval + 1)

        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e, time cost: %f" \
                                % (args.session, epoch, step, iters_per_epoch, loss_temp, lr, end-start))

        print("\t\t\timg_loss: %.4f, ins_loss: %.4f, cons_loss: %.4f" \
              % (img_loss.mean().item(), ins_loss.mean().item(), cons_loss.mean().item()))

        loss_temp = 0
        start = time.time()

    save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
    save_checkpoint({
      'session': args.session,
      'epoch': epoch + 1,
      'model': fasterRCNN_T.module.state_dict() if args.mGPUs else fasterRCNN_T.state_dict(),
      'optimizer': optimizer.state_dict(),
      'pooling_mode': cfg.POOLING_MODE,
      'class_agnostic': args.class_agnostic,
    }, save_name)
    print('save model: {}'.format(save_name))
    # args.beta-=0.1
    print(args.alpha, args.beta, args.w_cons, args.w_img, args.w_ins, args.lr_GD)

