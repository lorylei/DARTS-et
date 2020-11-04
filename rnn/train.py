import os
import gc
import sys
import glob
import time
import math
import numpy as np
import torch
import torch.nn as nn
import logging
import argparse
import genotypes
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import data
import model
import nltk
import torchtext
from torchtext.datasets import TranslationDataset
import pickle

from model import Network
from model import Baseline
from torch.autograd import Variable
import utils

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank/WikiText2 Language Model')
parser.add_argument('--data', type=str, default='../data/penn/',
                    help='location of the data corpus')
parser.add_argument('--emb_dim', type=int, default=512,
                    help='size of word embeddings')
parser.add_argument('--en_layers', type=int, default=3,
                    help='number of encoder layer')
parser.add_argument('--de_layers', type=int, default=3,
                    help='number of decoder layer')
# parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
# parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--learning_rate_min', type=float, default=1e-6, help='min learning rate')
parser.add_argument('--report_freq', type=float, default=500, help='report frequency')
parser.add_argument('--learning_rate', type=float, default=1e-4,
                    help='initial learning rate')
# parser.add_argument('--grad_clip', type=float, default=1,
#                     help='gradient clipping')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='batch size')
parser.add_argument('--searched', type=bool, default=True, help='use searched architecture?')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.75,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.25,
                    help='dropout for hidden nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropoutx', type=float, default=0.75,
                    help='dropout for input nodes rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.2,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1267,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='EXP',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1e-3,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=8e-7,
                    help='weight decay applied to all weights')
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from a checkpoint')
parser.add_argument('--small_batch_size', type=int, default=-1,
                    help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                     In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                     until batch_size is reached. An update step is then performed.')
parser.add_argument('--max_seq_len_delta', type=int, default=20,
                    help='max sequence length')
parser.add_argument('--single_gpu', default=True, action='store_false', 
                    help='use single GPU')
parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
args = parser.parse_args()


if not args.continue_train:
    args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
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
  logging.info(IWSLT2017.name)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  SRC = torchtext.data.Field(tokenize = "spacy",
                            tokenizer_language="en",
                            init_token = '<sos>',
                            eos_token = '<eos>',
                            lower = True)

  TRG = torchtext.data.Field(tokenize = "spacy",
                            tokenizer_language="de",
                            init_token = '<sos>',
                            eos_token = '<eos>',
                            lower = True)
  train_data, valid_data, test_data = IWSLT2017.splits(exts = ('_en.txt', '_de.txt'), fields = (SRC, TRG), root='../data')
  SRC.build_vocab(train_data, min_freq = 2)
  TRG.build_vocab(train_data, min_freq = 2)

  train_queue, valid_queue, test_queue = torchtext.data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = args.batch_size,
    device = torch.device('cuda'))

  PAD_IDX = TRG.vocab.stoi['<pad>']
  SOS_IDX = TRG.vocab.stoi['<sos>']
  EOS_IDX = TRG.vocab.stoi['<eos>']
  criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
  criterion = criterion.cuda()
  # genotype = {'gene_encoder': [{'combine_func': 'cat', 'left_activation': 'Swish', 'left_input': 1, 'left_layer': 'SA_h8', 'left_norm': 'batch_norm', 'right_activation': 'Swish', 'right_input': 0, 'right_layer': 'SA_h8', 'right_norm': 'batch_norm'}, {'combine_func': 'mul', 'left_activation': 'Swish', 'left_input': 2, 'left_layer': 'SA_h8', 'left_norm': 'none', 'right_activation': 'Swish', 'right_input': 2, 'right_layer': 'conv1d_1x1', 'right_norm': 'none'}, {'combine_func': 'add', 'left_activation': 'ReLU', 'left_input': 1, 'left_layer': 'identity', 'left_norm': 'batch_norm', 'right_activation': 'ReLU', 'right_input': 1, 'right_layer': 'GLU', 'right_norm': 'batch_norm'}, {'combine_func': 'add', 'left_activation': 'none', 'left_input': 4, 'left_layer': 'identity', 'left_norm': 'batch_norm', 'right_activation': 'none', 'right_input': 4, 'right_layer': 'identity', 'right_norm': 'batch_norm'}], 'gene_decoder': [{'combine_func': 'add', 'left_activation': 'ReLU', 'left_input': 0, 'left_layer': 'Att_En_h8', 'left_norm': 'batch_norm', 'right_activation': 'Swish', 'right_input': 0, 'right_layer': 'FFN_4', 'right_norm': 'none'}, {'combine_func': 'add', 'left_activation': 'none', 'left_input': 2, 'left_layer': 'Att_En_h8', 'left_norm': 'none', 'right_activation': 'none', 'right_input': 2, 'right_layer': 'Att_En_h8', 'right_norm': 'none'}, {'combine_func': 'add', 'left_activation': 'none', 'left_input': 3, 'left_layer': 'identity', 'left_norm': 'batch_norm', 'right_activation': 'ReLU', 'right_input': 3, 'right_layer': 'identity', 'right_norm': 'batch_norm'}, {'combine_func': 'mul', 'left_activation': 'ReLU', 'left_input': 4, 'left_layer': 'GLU', 'left_norm': 'none', 'right_activation': 'ReLU', 'right_input': 4, 'right_layer': 'conv1d_1x1', 'right_norm': 'none'}, {'combine_func': 'cat', 'left_activation': 'ReLU', 'left_input': 4, 'left_layer': 'conv1d_3x3', 'left_norm': 'none', 'right_activation': 'ReLU', 'right_input': 4, 'right_layer': 'conv1d_1x1', 'right_norm': 'none'}, {'combine_func': 'mul', 'left_activation': 'Swish', 'left_input': 4, 'left_layer': 'identity', 'left_norm': 'batch_norm', 'right_activation': 'ReLU', 'right_input': 3, 'right_layer': 'identity', 'right_norm': 'batch_norm'}]}
  # genotype search on Multi30k using first version of search space
  
  # genotype = {'gene_encoder': [{'combine_func': 'mul', 'left_activation': 'none', 'left_input': 1, 'left_layer': 'identity', 'norm': 'layer_norm', 'right_activation': 'none', 'right_input': 1, 'right_layer': 'identity'}, {'combine_func': 'add', 'left_activation': 'Swish', 'left_input': 2, 'left_layer': 'conv1d_1x1', 'norm': 'layer_norm', 'right_activation': 'none', 'right_input': 2, 'right_layer': 'GLU'}, {'combine_func': 'add', 'left_activation': 'ReLU', 'left_input': 3, 'left_layer': 'conv1d_3x3', 'norm': 'none', 'right_activation': 'ReLU', 'right_input': 3, 'right_layer': 'identity'}, {'combine_func': 'add', 'left_activation': 'Swish', 'left_input': 4, 'left_layer': 'identity', 'norm': 'layer_norm', 'right_activation': 'none', 'right_input': 0, 'right_layer': 'identity'}], 'gene_decoder': [{'combine_func': 'mul', 'left_activation': 'Swish', 'left_input': 0, 'left_layer': 'sep_conv_5x5', 'norm': 'none', 'right_activation': 'Swish', 'right_input': 0, 'right_layer': 'GLU'}, {'combine_func': 'add', 'left_activation': 'none', 'left_input': 2, 'left_layer': 'Att_En_h8', 'norm': 'none', 'right_activation': 'Swish', 'right_input': 2, 'right_layer': 'dead'}, {'combine_func': 'add', 'left_activation': 'Swish', 'left_input': 2, 'left_layer': 'identity', 'norm': 'layer_norm', 'right_activation': 'none', 'right_input': 3, 'right_layer': 'identity'}, {'combine_func': 'add', 'left_activation': 'none', 'left_input': 1, 'left_layer': 'identity', 'norm': 'layer_norm', 'right_activation': 'none', 'right_input': 4, 'right_layer': 'identity'}, {'combine_func': 'mul', 'left_activation': 'none', 'left_input': 5, 'left_layer': 'Att_En_h8', 'norm': 'layer_norm', 'right_activation': 'ReLU', 'right_input': 3, 'right_layer': 'Att_En_h8'}, {'combine_func': 'add', 'left_activation': 'Swish', 'left_input': 2, 'left_layer': 'dead', 'norm': 'layer_norm', 'right_activation': 'ReLU', 'right_input': 6, 'right_layer': 'identity'}]}
  # genotype search on NC2016 using second version of search space, and initialized with transformer, epoch 45

  genotype = {'gene_encoder': [{'combine_func': 'add', 'left_activation': 'LeakyReLU', 'left_input': 1, 'left_layer': 'sep_conv_7x7', 'norm': 'none', 'right_activation': 'none', 'right_input': 0, 'right_layer': 'sep_conv_7x7'}, {'combine_func': 'add', 'left_activation': 'ReLU', 'left_input': 1, 'left_layer': 'sep_conv_7x7', 'norm': 'none', 'right_activation': 'ReLU', 'right_input': 2, 'right_layer': 'conv1d_3x3'}, {'combine_func': 'add', 'left_activation': 'ReLU', 'left_input': 3, 'left_layer': 'sep_conv_7x7', 'norm': 'layer_norm', 'right_activation': 'none', 'right_input': 3, 'right_layer': 'SA_h8'}, {'combine_func': 'add', 'left_activation': 'Swish', 'left_input': 3, 'left_layer': 'sep_conv_7x7', 'norm': 'layer_norm', 'right_activation': 'none', 'right_input': 4, 'right_layer': 'identity'}], 'gene_decoder': [{'combine_func': 'add', 'left_activation': 'Swish', 'left_input': 0, 'left_layer': 'Att_En_h8', 'norm': 'layer_norm', 'right_activation': 'none', 'right_input': 0, 'right_layer': 'identity'}, {'combine_func': 'mul', 'left_activation': 'Swish', 'left_input': 0, 'left_layer': 'Att_En_h8', 'norm': 'none', 'right_activation': 'Swish', 'right_input': 2, 'right_layer': 'dead'}, {'combine_func': 'cat', 'left_activation': 'ReLU', 'left_input': 3, 'left_layer': 'Att_En_h8', 'norm': 'layer_norm', 'right_activation': 'none', 'right_input': 3, 'right_layer': 'GLU'}, {'combine_func': 'cat', 'left_activation': 'Swish', 'left_input': 3, 'left_layer': 'Att_En_h8', 'norm': 'layer_norm', 'right_activation': 'none', 'right_input': 4, 'right_layer': 'identity'}, {'combine_func': 'add', 'left_activation': 'Swish', 'left_input': 5, 'left_layer': 'Att_En_h8', 'norm': 'layer_norm', 'right_activation': 'none', 'right_input': 3, 'right_layer': 'Att_En_h8'}, {'combine_func': 'add', 'left_activation': 'none', 'left_input': 6, 'left_layer': 'FFN_4', 'norm': 'layer_norm', 'right_activation': 'none', 'right_input': 6, 'right_layer': 'identity'}]}
  # genotype search on NC2016sampled using second version of search space, and initialized with tranformer, epoch 49

  if args.searched:
    model = Network(args.emb_dim, len(SRC.vocab), len(TRG.vocab), args.en_layers, args.de_layers, criterion, SRC.vocab.stoi['<pad>'], genotype, SOS_IDX, EOS_IDX)
  else:
    model = Baseline(args.emb_dim, len(SRC.vocab), len(TRG.vocab), args.en_layers, args.de_layers, criterion, SRC.vocab.stoi['<pad>'], genotype, SOS_IDX, EOS_IDX)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  # optimizer = torch.optim.SGD(
  #     model.parameters(),
  #     args.learning_rate,
  #     momentum=args.momentum,
  #     weight_decay=args.weight_decay)

  optimizer = torch.optim.Adam(
      model.parameters(),
      args.learning_rate,
      betas=(0.9, 0.999),
      eps=1e-7)


  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)


  train_loss_lis = []
  valid_loss_lis = []
  best_loss = 1000

  try:
    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)


        # print(F.softmax(model.alphas_normal, dim=-1))
        # print(F.softmax(model.alphas_reduce, dim=-1))

        # training
        train_loss = train(train_queue, model, criterion, optimizer, lr)
        train_loss_lis.append(train_loss)

        # validation
        valid_loss = validate(valid_queue, model, criterion)
        valid_loss_lis.append(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            utils.save(model, os.path.join(args.save, 'weights.pt'))
        fw = open(os.path.join(args.save, 'loss'), 'wb')
        pickle.dump((train_loss_lis, valid_loss_lis), fw, -1)
        fw.close()
  except KeyboardInterrupt:
    logging.info('Exiting from training early')

  logging.info('loading best model...')
  state = torch.load(os.path.join(args.save, 'weights.pt'))#, map_location='cuda'
  if args.searched:
    model = Network(args.emb_dim, len(SRC.vocab), len(TRG.vocab), args.en_layers, args.de_layers, criterion, SRC.vocab.stoi['<pad>'], genotype, SOS_IDX, EOS_IDX)
  else:
    model = Baseline(args.emb_dim, len(SRC.vocab), len(TRG.vocab), args.en_layers, args.de_layers, criterion, SRC.vocab.stoi['<pad>'], genotype, SOS_IDX, EOS_IDX)
  model.load_state_dict(state)
  model = model.cuda()
  logging.info('testing...')
  bleu_score = infer(test_queue, model, TRG)
  logging.info('bleu score: %f', bleu_score)


def train(train_queue, model, criterion, optimizer, lr):
  objs = utils.AvgrageMeter()

  # stop = len(train_queue)//50

  for step, batch in enumerate(train_queue):
    # if step > stop:
    #   break

    input = batch.src.cuda()
    target = batch.trg.cuda(non_blocking=True)
    model.train()
    n = target.size(0)-1

    optimizer.zero_grad()
    loss = model._loss(input, target)

    objs.update(loss.item(), n)
    loss.backward()
    # nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    if step % args.report_freq == 0:
      logging.info('train %03d %f', step, objs.avg)

  return objs.avg


def validate(valid_queue, model, criterion):
  model.eval()
  objs = utils.AvgrageMeter()

  with torch.no_grad():

    for step, batch in enumerate(valid_queue):
        input = batch.src.cuda()
        target = batch.trg.cuda(non_blocking=True)
        n = target.size(0)-1

        loss = model._loss(input, target)

        objs.update(loss.item(), n)
    logging.info('valid %f', objs.avg)

    return objs.avg

def convert_to_sent(ids_tensor, TRG, ref=False):
  EOS_IDX = TRG.vocab.stoi['<eos>']
  ids = ids_tensor.t().contiguous().cpu().tolist()
  batch_ids = []
  for sent_ids in ids:
    sent_item = []
    for token_id in sent_ids:
      sent_item.append(TRG.vocab.itos[token_id])
      if token_id==EOS_IDX:
        break
    if ref:
      batch_ids.append([sent_item,])
    else:
      batch_ids.append(sent_item)

  return batch_ids
      

def infer(test_queue, model, TRG):

  with torch.no_grad():
    hypo = []
    ref = []
    for batch in test_queue:
      input = batch.src.cuda()
      output = model.decode(input)
      hypo += convert_to_sent(output, TRG)
      ref += convert_to_sent(batch.trg, TRG, ref=True)

    score = nltk.translate.bleu_score.corpus_bleu(ref, hypo)

    logging.info('number of sentences:', len(hypo))
    logging.info( str(list(zip(hypo[50:100], ref[50:100]))))
    return score
          



if __name__ == '__main__':
  main() 
