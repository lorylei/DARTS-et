import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchtext
from torchtext.datasets import TranslationDataset
import torch.backends.cudnn as cudnn
import pickle
from torch.nn.parallel import DistributedDataParallel

import spacy

from torch.autograd import Variable
from model_search import Network
from architect import Architect


parser = argparse.ArgumentParser("WMT14")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=24, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--emb_dim', type=int, default=128, help='num of init channels')
parser.add_argument('--en_layers', type=int, default=3, help='total number of layers')
parser.add_argument('--de_layers', type=int, default=3, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

class IWSLT2017(TranslationDataset):

    name = 'NC2016'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='.data',
               train='concatenated_en2de_train', validation='concatenated_en2de_dev', test='concatenated_en2de_test', **kwargs):
        """Create dataset objects for splits of the Multi30k dataset.

        Arguments:
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            root: Root dataset storage directory. Default is '.data'.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """

        # TODO: This is a _HORRIBLE_ patch related to #208
        # 'path' can be passed as a kwarg to the translation dataset constructor
        # or has to be set (so the download wouldn't be duplicated). A good idea
        # seems to rename the existence check variable from path to something else
        if 'path' not in kwargs:
            expected_folder = os.path.join(root, cls.name)
            path = expected_folder if os.path.exists(expected_folder) else None
        else:
            path = kwargs['path']
            del kwargs['path']

        return super(IWSLT2017, cls).splits(
            exts, fields, path, root, train, validation, test, **kwargs)

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  SRC = torchtext.data.Field(tokenize = "spacy",
                            tokenizer_language="en_core_web_sm",
                            init_token = '<sos>',
                            eos_token = '<eos>',
                            lower = True)

  TRG = torchtext.data.Field(tokenize = "spacy",
                            tokenizer_language="de_core_news_sm",
                            init_token = '<sos>',
                            eos_token = '<eos>',
                            lower = True)
  # train_data, valid_data, test_data = torchtext.datasets.Multi30k.splits(exts = ('.de', '.en'), fields = (SRC, TRG), root='../data')
  train_data, valid_data, test_data = IWSLT2017.splits(exts = ('_en.txt', '_de.txt'), fields = (SRC, TRG), root='../data')
  SRC.build_vocab(train_data, min_freq = 2)
  TRG.build_vocab(train_data, min_freq = 2)

  train_queue, valid_queue, test_queue = torchtext.data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = args.batch_size,
    device = torch.device('cuda'))

  PAD_IDX = TRG.vocab.stoi['<pad>']
  criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
  criterion = criterion.cuda()
  model = Network(args.emb_dim, len(SRC.vocab), len(TRG.vocab), args.en_layers, args.de_layers, criterion, SRC.vocab.stoi['<pad>'])
  torch.distributed.init_process_group(backend="nccl")
  model = DistributedDataParallel(model.cuda(), find_unused_parameters=True) # device_ids will include all GPU devices by default
  # model = nn.DataParallel(model.cuda())
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.module.parameters(filter = True),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)


  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  train_loss_lis = []
  valid_loss_lis = []

  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.module.genotype()
    logging.info('genotype = %s', genotype)

    # print(F.softmax(model.alphas_normal, dim=-1))
    # print(F.softmax(model.alphas_reduce, dim=-1))
    logging.info(str(model.module.alphas_en.return_weights(prob=True)))
    logging.info(str(model.module.alphas_de.return_weights(prob=True)))

    # training
    train_loss = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)
    train_loss_lis.append(train_loss)

    # validation
    valid_loss = infer(valid_queue, model, criterion)
    valid_loss_lis.append(valid_loss)

    utils.save(model, os.path.join(args.save, 'weights.pt'))
    fw = open(os.path.join(args.save, 'loss'), 'wb')
    pickle.dump((train_loss_lis, valid_loss_lis), fw, -1)
    fw.close()


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
  objs = utils.AvgrageMeter()
  # print(len(train_queue))
  for step, batch in enumerate(train_queue):
    # denominator = 10
    # stop = len(train_queue)//50
    # if step > stop:
    #   break
    input = batch.src
    target = batch.trg
    model.train()
    n = target.size(0)-1

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(non_blocking =True)

    # get a random minibatch from the search queue with replacement
    batch_search = next(iter(valid_queue))
    input_search, target_search = batch_search.src, batch_search.trg
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda(non_blocking =True)

    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    loss = model.module._loss(input, target, model)

    objs.update(loss.item(), n)
    loss.backward()
    nn.utils.clip_grad_norm(model.module.parameters(filter = True), args.grad_clip)
    optimizer.step()

    if step % args.report_freq == 0:
      logging.info('train %03d %f', step, objs.avg)

  return objs.avg


def infer(valid_queue, model, criterion):
  model.eval()
  objs = utils.AvgrageMeter()

  for step, batch in enumerate(valid_queue):
    input, target = batch.src, batch.trg
    n = target.size(0)-1
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(non_blocking =True)

    loss = model.module._loss(input, target, model)

    objs.update(loss.item(), n)
    if step % args.report_freq == 0:
      logging.info('valid %03d %f', step, objs.avg)

  return objs.avg


if __name__ == '__main__':
  main() 

