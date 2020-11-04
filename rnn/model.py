import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
import math

# genotype = {'gene_encoder': [{'combine_func': 'cat', 'left_activation': 'Swish', 'left_input': 1, 'left_layer': 'SA_h8', 'left_norm': 'batch_norm', 'right_activation': 'Swish', 'right_input': 0, 'right_layer': 'SA_h8', 'right_norm': 'batch_norm'}, {'combine_func': 'mul', 'left_activation': 'Swish', 'left_input': 2, 'left_layer': 'SA_h8', 'left_norm': 'none', 'right_activation': 'Swish', 'right_input': 2, 'right_layer': 'conv1d_1x1', 'right_norm': 'none'}, {'combine_func': 'add', 'left_activation': 'ReLU', 'left_input': 1, 'left_layer': 'identity', 'left_norm': 'batch_norm', 'right_activation': 'ReLU', 'right_input': 1, 'right_layer': 'GLU', 'right_norm': 'batch_norm'}, {'combine_func': 'add', 'left_activation': 'none', 'left_input': 4, 'left_layer': 'identity', 'left_norm': 'batch_norm', 'right_activation': 'none', 'right_input': 4, 'right_layer': 'identity', 'right_norm': 'batch_norm'}], 'gene_decoder': [{'combine_func': 'add', 'left_activation': 'ReLU', 'left_input': 0, 'left_layer': 'Att_En_h8', 'left_norm': 'batch_norm', 'right_activation': 'Swish', 'right_input': 0, 'right_layer': 'FFN_4', 'right_norm': 'none'}, {'combine_func': 'add', 'left_activation': 'none', 'left_input': 2, 'left_layer': 'Att_En_h8', 'left_norm': 'none', 'right_activation': 'none', 'right_input': 2, 'right_layer': 'Att_En_h8', 'right_norm': 'none'}, {'combine_func': 'add', 'left_activation': 'none', 'left_input': 3, 'left_layer': 'identity', 'left_norm': 'batch_norm', 'right_activation': 'ReLU', 'right_input': 3, 'right_layer': 'identity', 'right_norm': 'batch_norm'}, {'combine_func': 'mul', 'left_activation': 'ReLU', 'left_input': 4, 'left_layer': 'GLU', 'left_norm': 'none', 'right_activation': 'ReLU', 'right_input': 4, 'right_layer': 'conv1d_1x1', 'right_norm': 'none'}, {'combine_func': 'cat', 'left_activation': 'ReLU', 'left_input': 4, 'left_layer': 'conv1d_3x3', 'left_norm': 'none', 'right_activation': 'ReLU', 'right_input': 4, 'right_layer': 'conv1d_1x1', 'right_norm': 'none'}, {'combine_func': 'mul', 'left_activation': 'Swish', 'left_input': 4, 'left_layer': 'identity', 'left_norm': 'batch_norm', 'right_activation': 'ReLU', 'right_input': 3, 'right_layer': 'identity', 'right_norm': 'batch_norm'}]}

class TwoBranchOp(nn.Module):

  def __init__(self, C, gene, decode=False):
    super(TwoBranchOp, self).__init__()
    self.decode = decode
    self.left_input = gene['left_input']
    self.right_input = gene['right_input']

    # self._norm_ops_l = Norm_OPS[gene['left_norm']](C, False)
    # self._norm_ops_r = Norm_OPS[gene['right_norm']](C, False)

    if decode:
      Layer_OPS = De_Layer_OPS
    else:
      Layer_OPS = En_Layer_OPS
    self._layer_ops_l = Layer_OPS[gene['left_layer']](C)
    self._layer_ops_r = Layer_OPS[gene['right_layer']](C)

    self._act_ops_l = Act_OPS[gene['left_activation']]()
    self._act_ops_r = Act_OPS[gene['right_activation']]()

    self._cmb_ops = Cmb_OPS[gene['combine_func']](C)
    self._norm_ops = Norm_OPS[gene['norm']](C, False)

  def forward(self, x, mask, memory=None):  #weights: dict {left_input:[], left_norm:[], left_layer:[], left_activation:[], right.... combine_func:[]}
    left_x = x[self.left_input]
    # left_x = self._norm_ops_l(left_x)
    if self.decode:
      left_x = self._layer_ops_l(left_x, mask, memory)
    else:
      left_x = self._layer_ops_l(left_x, mask)
    left_x = self._act_ops_l(left_x)

    right_x = x[self.right_input]
    # right_x = self._norm_ops_r(right_x)
    if self.decode:
      right_x = self._layer_ops_r(right_x, mask, memory)
    else:
      right_x = self._layer_ops_r(right_x, mask)
    right_x = self._act_ops_r(right_x)

    x = self._cmb_ops(left_x, right_x)
    x = self._norm_ops(x)

    return x
    

class EvolvedCell(nn.Module):

  def __init__(self, C, genotype, decode=False):
    super(EvolvedCell, self).__init__()
    
    if decode:
      gene = genotype['gene_decoder']
    else:
      gene = genotype['gene_encoder']
    self._steps = len(gene)

    self._ops = nn.ModuleList()
    for i in range(self._steps):
      op = TwoBranchOp(C, gene[i], decode)
      self._ops.append(op)

  def forward(self, s0, s1, mask, memory=None):

    states = [s0, s1]
    for i in range(self._steps):
      s = self._ops[i](states, mask, memory)
      states.append(s)

    return states[-1]

class PositionalEncoding(nn.Module):

  def __init__(self, d_model, dropout=0.1, max_len=5000):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + self.pe[:x.size(0), :]
    return self.dropout(x)

class Network(nn.Module):

  def __init__(self, C, input_vocab, output_vocab, en_layers, de_layers, criterion, padding_idx, genotype, sos_idx, eos_idx):
    super(Network, self).__init__()
    self._C = C
    self._input_vocab = input_vocab
    self._output_vocab = output_vocab
    self._en_layers = en_layers
    self._de_layers = de_layers
    self._criterion = criterion
    self._pad_idx = padding_idx
    self._sos_idx = sos_idx
    self._eos_idx = eos_idx

    self.embed_input = nn.Embedding(input_vocab, C)
    self.dropout = nn.Dropout(0.1)
    self.pos_encoder = PositionalEncoding(C)
 
    self.en_cells = nn.ModuleList()
    for i in range(en_layers):
      cell = EvolvedCell(C, genotype, decode=False)
      self.en_cells += [cell]

    self.embed_target = nn.Embedding(output_vocab, C)
    self.de_cells = nn.ModuleList()
    for i in range(de_layers):
      cell = EvolvedCell(C, genotype, decode=True)
      self.de_cells += [cell]

    self.classifier = nn.Linear(C, output_vocab)

    # self._initialize_alphas()

  def forward(self, input, target):
    mask = (input == self._pad_idx).t().contiguous()
    # print('padding idx:', self._pad_idx)
    # print(input)
    # print(target)
    input = self.dropout(self.embed_input(input))
    input = self.pos_encoder(input)
    s0 = s1 = input

    for i, cell in enumerate(self.en_cells):
      s0, s1 = s1, cell(s0, s1, mask)
    memory = s1

    target = self.dropout(self.embed_target(target))
    target = self.pos_encoder(target)
    s0 = s1 = target

    for i, cell in enumerate(self.de_cells):
      s0, s1 = s1, cell(s0, s1, mask, memory)

    logits = self.classifier(s1)
    logits = logits.view(-1, logits.size(-1))
    return logits

  def _loss(self, input, target):
    logits = self(input, target[:-1])
    return self._criterion(logits, target[1:].view(-1)) 

  def decode(self, input, max_len=50):
    mask = (input == self._pad_idx).t().contiguous()
    batch_size = input.size(1)
    device = input.device
    # print('padding idx:', self._pad_idx)
    # print(input)
    # print(target)
    input = self.dropout(self.embed_input(input))
    input = self.pos_encoder(input)
    s0 = s1 = input

    for i, cell in enumerate(self.en_cells):
      s0, s1 = s1, cell(s0, s1, mask)
    memory = s1

    input_token = torch.full((1,batch_size), self._sos_idx, dtype=torch.long, device=device)
    has_ended = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for i in range(max_len):
      target = self.dropout(self.embed_target(input_token))
      target = self.pos_encoder(target)
      s0 = s1 = target

      for i, cell in enumerate(self.de_cells):
        s0, s1 = s1, cell(s0, s1, mask, memory)

      logits = self.classifier(s1)
      next_token = torch.argmax(logits[-1], dim=-1)
      input_token = torch.cat((input_token, next_token.unsqueeze(0)), dim=0)
      has_ended = has_ended | (next_token == self._eos_idx)
      if has_ended.sum().item() == has_ended.size(0):
        break
    
    return input_token

class Baseline(nn.Module):

  def __init__(self, C, input_vocab, output_vocab, en_layers, de_layers, criterion, padding_idx, genotype, sos_idx, eos_idx):
    super(Baseline, self).__init__()
    self._C = C
    self._input_vocab = input_vocab
    self._output_vocab = output_vocab
    self._en_layers = en_layers
    self._de_layers = de_layers
    self._criterion = criterion
    self._pad_idx = padding_idx
    self._sos_idx = sos_idx
    self._eos_idx = eos_idx

    self.embed_input = nn.Embedding(input_vocab, C)
    self.dropout = nn.Dropout(0.1)
    self.pos_encoder = PositionalEncoding(C)
  
    encoder_layer = nn.TransformerEncoderLayer(C, 8)
    self.encoder = nn.TransformerEncoder(encoder_layer, en_layers)

    self.embed_target = nn.Embedding(output_vocab, C)

    decoder_layer = nn.TransformerDecoderLayer(C, 8)
    self.decoder = nn.TransformerDecoder(decoder_layer, de_layers)

    self.classifier = nn.Linear(C, output_vocab)


  def forward(self, input, target):
    mask = (input == self._pad_idx).t().contiguous()
    device = target.device
    decode_mask = torch.triu( torch.ones(len(target), len(target)) ).t().to(device) == 0
    # print('padding idx:', self._pad_idx)
    # print(input)
    # print(target)
    input = self.dropout(self.embed_input(input))
    input = self.pos_encoder(input)
 
    memory = self.encoder(input, src_key_padding_mask=mask)

    target = self.dropout(self.embed_target(target))
    target = self.pos_encoder(target)

    target = self.decoder(target, memory, tgt_mask=decode_mask, memory_key_padding_mask=mask)

    logits = self.classifier(target)
    logits = logits.view(-1, logits.size(-1))
    return logits

  def _loss(self, input, target):
    logits = self(input, target[:-1])
    return self._criterion(logits, target[1:].view(-1)) 

  def decode(self, input, max_len=50):
    mask = (input == self._pad_idx).t().contiguous()
    batch_size = input.size(1)
    device = input.device
    # print('padding idx:', self._pad_idx)
    # print(input)
    # print(target)
    input = self.dropout(self.embed_input(input))
    input = self.pos_encoder(input)
 
    memory = self.encoder(input, src_key_padding_mask=mask)

    input_token = torch.full((1,batch_size), self._sos_idx, dtype=torch.long, device=device)
    has_ended = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for i in range(max_len):
      decode_mask = torch.triu( torch.ones(len(input_token), len(input_token)) ).t().to(device) == 0
      target = self.dropout(self.embed_target(input_token))
      target = self.pos_encoder(target)

      target = self.decoder(target, memory, tgt_mask=decode_mask, memory_key_padding_mask=mask)

      logits = self.classifier(target)
      next_token = torch.argmax(logits[-1], dim=-1)
      input_token = torch.cat((input_token, next_token.unsqueeze(0)), dim=0)
      has_ended = has_ended | (next_token == self._eos_idx)
      if has_ended.sum().item() == has_ended.size(0):
        break
    
    return input_token


# genotype = {'gene_encoder': [{'combine_func': 'cat', 'left_activation': 'Swish', 'left_input': 1, 'left_layer': 'SA_h8', 'left_norm': 'batch_norm', 'right_activation': 'Swish', 'right_input': 0, 'right_layer': 'SA_h8', 'right_norm': 'batch_norm'}, {'combine_func': 'mul', 'left_activation': 'Swish', 'left_input': 2, 'left_layer': 'SA_h8', 'left_norm': 'none', 'right_activation': 'Swish', 'right_input': 2, 'right_layer': 'conv1d_1x1', 'right_norm': 'none'}, {'combine_func': 'add', 'left_activation': 'ReLU', 'left_input': 1, 'left_layer': 'identity', 'left_norm': 'batch_norm', 'right_activation': 'ReLU', 'right_input': 1, 'right_layer': 'GLU', 'right_norm': 'batch_norm'}, {'combine_func': 'add', 'left_activation': 'none', 'left_input': 4, 'left_layer': 'identity', 'left_norm': 'batch_norm', 'right_activation': 'none', 'right_input': 4, 'right_layer': 'identity', 'right_norm': 'batch_norm'}], 'gene_decoder': [{'combine_func': 'add', 'left_activation': 'ReLU', 'left_input': 0, 'left_layer': 'Att_En_h8', 'left_norm': 'batch_norm', 'right_activation': 'Swish', 'right_input': 0, 'right_layer': 'FFN_4', 'right_norm': 'none'}, {'combine_func': 'add', 'left_activation': 'none', 'left_input': 2, 'left_layer': 'Att_En_h8', 'left_norm': 'none', 'right_activation': 'none', 'right_input': 2, 'right_layer': 'Att_En_h8', 'right_norm': 'none'}, {'combine_func': 'add', 'left_activation': 'none', 'left_input': 3, 'left_layer': 'identity', 'left_norm': 'batch_norm', 'right_activation': 'ReLU', 'right_input': 3, 'right_layer': 'identity', 'right_norm': 'batch_norm'}, {'combine_func': 'mul', 'left_activation': 'ReLU', 'left_input': 4, 'left_layer': 'GLU', 'left_norm': 'none', 'right_activation': 'ReLU', 'right_input': 4, 'right_layer': 'conv1d_1x1', 'right_norm': 'none'}, {'combine_func': 'cat', 'left_activation': 'ReLU', 'left_input': 4, 'left_layer': 'conv1d_3x3', 'left_norm': 'none', 'right_activation': 'ReLU', 'right_input': 4, 'right_layer': 'conv1d_1x1', 'right_norm': 'none'}, {'combine_func': 'mul', 'left_activation': 'Swish', 'left_input': 4, 'left_layer': 'identity', 'left_norm': 'batch_norm', 'right_activation': 'ReLU', 'right_input': 3, 'right_layer': 'identity', 'right_norm': 'batch_norm'}]}
