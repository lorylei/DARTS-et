import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
import math

class TwoBranchOp(nn.Module):

  def __init__(self, C, decode=False):
    super(TwoBranchOp, self).__init__()
    self.decode = decode
    self._norm_ops = nn.ModuleList()
    # self._norm_ops_r = nn.ModuleList()
    for key in Norm_OPS:
      self._norm_ops.append(Norm_OPS[key](C, False))
      # self._norm_ops_r.append(Norm_OPS[key](C, False))

    self._layer_ops_l = nn.ModuleList()
    self._layer_ops_r = nn.ModuleList()
    if decode:
      Layer_OPS = De_Layer_OPS
    else:
      Layer_OPS = En_Layer_OPS
    for key in Layer_OPS:
      self._layer_ops_l.append(Layer_OPS[key](C))
      self._layer_ops_r.append(Layer_OPS[key](C))

    self._act_ops_l = nn.ModuleList()
    self._act_ops_r = nn.ModuleList()
    for key in Act_OPS:
      self._act_ops_l.append(Act_OPS[key]())
      self._act_ops_r.append(Act_OPS[key]())

    self._cmb_ops = nn.ModuleList()
    for key in Cmb_OPS:
      self._cmb_ops.append(Cmb_OPS[key](C))

  def forward(self, x, weights, mask, memory=None):  #weights: dict {left_input:[], left_norm:[], left_layer:[], left_activation:[], right.... combine_func:[]}
    x=torch.cat( list(map(lambda x: x.unsqueeze(0), x)) ,dim=0)
    left_x=torch.sum( x * weights['left_input'].view(-1,1,1,1) , dim=0)
    # left_x=sum(w * op(left_x) for w, op in zip(weights['left_norm'], self._norm_ops_l))
    if self.decode:
      left_x=sum(w * op(left_x, mask, memory) for w, op in zip(weights['left_layer'], self._layer_ops_l))
    else:
      left_x=sum(w * op(left_x, mask) for w, op in zip(weights['left_layer'], self._layer_ops_l))
    left_x=sum(w * op(left_x) for w, op in zip(weights['left_activation'], self._act_ops_l))

    right_x=torch.sum( x * weights['right_input'].view(-1,1,1,1) , dim=0)
    # right_x=sum(w * op(right_x) for w, op in zip(weights['right_norm'], self._norm_ops_r))
    if self.decode:
      right_x=sum(w * op(right_x, mask, memory) for w, op in zip(weights['right_layer'], self._layer_ops_r))
    else:
      right_x=sum(w * op(right_x, mask) for w, op in zip(weights['right_layer'], self._layer_ops_r))
    right_x=sum(w * op(right_x) for w, op in zip(weights['right_activation'], self._act_ops_r))

    x=sum(w * op(left_x, right_x) for w, op in zip(weights['combine_func'], self._cmb_ops))
    x=sum(w * op(x) for w, op in zip(weights['norm'], self._norm_ops))

    return x
    

class EvolvedCell(nn.Module):

  def __init__(self, C, steps, decode=False):
    super(EvolvedCell, self).__init__()
    
    self._steps = steps

    self._ops = nn.ModuleList()
    for i in range(self._steps):
      op = TwoBranchOp(C, decode)
      self._ops.append(op)

  def forward(self, s0, s1, weights, mask, memory=None):

    states = [s0, s1]
    for i in range(self._steps):
      s = self._ops[i](states, weights[i], mask, memory)
      states.append(s)

    return states[-1]


class BlockWeights(nn.Module):
  def __init__(self, input_num, decode):
    super(BlockWeights, self).__init__()

    def init_ops(OPS_set, OPS, multiplier=3.0):
      if type(OPS) == str:
        weights = torch.zeros(len(OPS_set),)
        for i, key in enumerate(OPS_set):
          if key == OPS:
            weights[i] = multiplier
            break
      elif type(OPS) == int:
        weights = torch.zeros(OPS_set,)
        weights[OPS] = multiplier
      return weights

    if decode:
      Layer_OPS = De_Layer_OPS
      if input_num % 3 == 2:
        self.param = nn.ParameterDict({'left_input': nn.Parameter( init_ops(input_num, input_num-1) ),
                                    #  'left_norm': nn.Parameter( 1e-3*torch.randn(len(Norm_OPS),) ),
                                    'left_layer': nn.Parameter( init_ops(Layer_OPS, 'SA_h8') ),
                                    'left_activation': nn.Parameter( init_ops(Act_OPS, 'none') ),

                                    'right_input': nn.Parameter( init_ops(input_num, input_num-1) ),
                                    #  'right_norm': nn.Parameter( 1e-3*torch.randn(len(Norm_OPS),) ),
                                    'right_layer': nn.Parameter( init_ops(Layer_OPS, 'identity') ),
                                    'right_activation': nn.Parameter( init_ops(Act_OPS, 'none') ),

                                    'combine_func': nn.Parameter( init_ops(Cmb_OPS, 'add') ),
                                    'norm': nn.Parameter( init_ops(Norm_OPS, 'layer_norm') ),
                                      })
      elif input_num % 3 == 0:
        self.param = nn.ParameterDict({'left_input': nn.Parameter( init_ops(input_num, input_num-1) ),
                                    #  'left_norm': nn.Parameter( 1e-3*torch.randn(len(Norm_OPS),) ),
                                    'left_layer': nn.Parameter( init_ops(Layer_OPS, 'Att_En_h8') ),
                                    'left_activation': nn.Parameter( init_ops(Act_OPS, 'none') ),

                                    'right_input': nn.Parameter( init_ops(input_num, input_num-1) ),
                                    #  'right_norm': nn.Parameter( 1e-3*torch.randn(len(Norm_OPS),) ),
                                    'right_layer': nn.Parameter( init_ops(Layer_OPS, 'identity') ),
                                    'right_activation': nn.Parameter( init_ops(Act_OPS, 'none') ),

                                    'combine_func': nn.Parameter( init_ops(Cmb_OPS, 'add') ),
                                    'norm': nn.Parameter( init_ops(Norm_OPS, 'layer_norm') ),
                                      })
      else:
        self.param = nn.ParameterDict({'left_input': nn.Parameter( init_ops(input_num, input_num-1) ),
                                    #  'left_norm': nn.Parameter( 1e-3*torch.randn(len(Norm_OPS),) ),
                                    'left_layer': nn.Parameter( init_ops(Layer_OPS, 'FFN_4') ),
                                    'left_activation': nn.Parameter( init_ops(Act_OPS, 'none') ),

                                    'right_input': nn.Parameter( init_ops(input_num, input_num-1) ),
                                    #  'right_norm': nn.Parameter( 1e-3*torch.randn(len(Norm_OPS),) ),
                                    'right_layer': nn.Parameter( init_ops(Layer_OPS, 'identity') ),
                                    'right_activation': nn.Parameter( init_ops(Act_OPS, 'none') ),

                                    'combine_func': nn.Parameter( init_ops(Cmb_OPS, 'add') ),
                                    'norm': nn.Parameter( init_ops(Norm_OPS, 'layer_norm') ),
                                      })

    else:
      Layer_OPS = En_Layer_OPS
      if input_num % 2 == 0:
        self.param = nn.ParameterDict({'left_input': nn.Parameter( init_ops(input_num, input_num-1) ),
                                    #  'left_norm': nn.Parameter( 1e-3*torch.randn(len(Norm_OPS),) ),
                                    'left_layer': nn.Parameter( init_ops(Layer_OPS, 'SA_h8') ),
                                    'left_activation': nn.Parameter( init_ops(Act_OPS, 'none') ),

                                    'right_input': nn.Parameter( init_ops(input_num, input_num-1) ),
                                    #  'right_norm': nn.Parameter( 1e-3*torch.randn(len(Norm_OPS),) ),
                                    'right_layer': nn.Parameter( init_ops(Layer_OPS, 'identity') ),
                                    'right_activation': nn.Parameter( init_ops(Act_OPS, 'none') ),

                                    'combine_func': nn.Parameter( init_ops(Cmb_OPS, 'add') ),
                                    'norm': nn.Parameter( init_ops(Norm_OPS, 'layer_norm') ),
                                      })
      else:
        self.param = nn.ParameterDict({'left_input': nn.Parameter( init_ops(input_num, input_num-1) ),
                                    #  'left_norm': nn.Parameter( 1e-3*torch.randn(len(Norm_OPS),) ),
                                    'left_layer': nn.Parameter( init_ops(Layer_OPS, 'FFN_4') ),
                                    'left_activation': nn.Parameter( init_ops(Act_OPS, 'none') ),

                                    'right_input': nn.Parameter( init_ops(input_num, input_num-1) ),
                                    #  'right_norm': nn.Parameter( 1e-3*torch.randn(len(Norm_OPS),) ),
                                    'right_layer': nn.Parameter( init_ops(Layer_OPS, 'identity') ),
                                    'right_activation': nn.Parameter( init_ops(Act_OPS, 'none') ),

                                    'combine_func': nn.Parameter( init_ops(Cmb_OPS, 'add') ),
                                    'norm': nn.Parameter( init_ops(Norm_OPS, 'layer_norm') ),
                                      })
    # self.param = nn.ParameterDict({'left_input': nn.Parameter( 1e-3*torch.randn(input_num,) ),
    #                               #  'left_norm': nn.Parameter( 1e-3*torch.randn(len(Norm_OPS),) ),
    #                                'left_layer': nn.Parameter( 1e-3*torch.randn(len(Layer_OPS),) ),
    #                                'left_activation': nn.Parameter( 1e-3*torch.randn(len(Act_OPS),) ),

    #                                'right_input': nn.Parameter( 1e-3*torch.randn(input_num,) ),
    #                               #  'right_norm': nn.Parameter( 1e-3*torch.randn(len(Norm_OPS),) ),
    #                                'right_layer': nn.Parameter( 1e-3*torch.randn(len(Layer_OPS),) ),
    #                                'right_activation': nn.Parameter( 1e-3*torch.randn(len(Act_OPS),) ),

    #                                'combine_func': nn.Parameter( 1e-3*torch.randn(len(Cmb_OPS),) ),
    #                                'norm': nn.Parameter( 1e-3*torch.randn(len(Norm_OPS),) ),
    #                                 })
    # print(self.param)

  def return_weights(self, prob=False):
    reg_weights={}
    # print(self.param)
    for key in self.param:
      if prob:
        reg_weights[key]=F.softmax( self.param[key], dim=0).data
      else:
        reg_weights[key]=F.softmax( self.param[key], dim=0)
        # reg_weights[key]=F.gumbel_softmax( self.param[key], hard=True, dim=0)
    return reg_weights


class CellWeights(nn.Module):
  def __init__(self, steps, decode=False):
    super(CellWeights, self).__init__()
    self.param = nn.ModuleList()
    for i in range(steps):
      self.param.append(BlockWeights(i+2, decode))

  def return_weights(self, prob=False):
    reg_weights=[]
    for block in self.param:
      reg_weights.append( block.return_weights(prob) )
    return reg_weights

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

  def __init__(self, C, input_vocab, output_vocab, en_layers, de_layers, criterion, padding_idx, en_steps=4, de_steps=6):
    super(Network, self).__init__()
    self._C = C
    self._input_vocab = input_vocab
    self._output_vocab = output_vocab
    self._en_layers = en_layers
    self._de_layers = de_layers
    self._criterion = criterion
    self._en_steps = en_steps
    self._de_steps = de_steps
    self._pad_idx = padding_idx

    self.embed_input = nn.Embedding(input_vocab, C)
    self.dropout = nn.Dropout(0.1)
    self.pos_encoder = PositionalEncoding(C)
 
    self.en_cells = nn.ModuleList()
    for i in range(en_layers):
      cell = EvolvedCell(C, en_steps, decode=False)
      self.en_cells += [cell]

    self.embed_target = nn.Embedding(output_vocab, C)
    self.de_cells = nn.ModuleList()
    for i in range(de_layers):
      cell = EvolvedCell(C, de_steps, decode=True)
      self.de_cells += [cell]

    self.classifier = nn.Linear(C, output_vocab)

    # self.alphas = [CellWeights(en_steps).cuda(), CellWeights(de_steps, True).cuda()]
    self.alphas_en = CellWeights(en_steps)
    self.alphas_de = CellWeights(de_steps, True)
    # self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._input_vocab, self._output_vocab, self._en_layers, self._de_layers, self._criterion, self._pad_idx).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input, target):
    mask = (input == self._pad_idx).t().contiguous()
    # print('padding idx:', self._pad_idx)
    # print(input)
    # print(target)
    input = self.dropout(self.embed_input(input))
    input = self.pos_encoder(input)
    s0 = s1 = input
    weights = self.alphas_en.return_weights()
    # print(weights)
    for i, cell in enumerate(self.en_cells):
      s0, s1 = s1, cell(s0, s1, weights, mask)
    memory = s1

    target = self.dropout(self.embed_target(target))
    target = self.pos_encoder(target)
    s0 = s1 = target
    weights = self.alphas_de.return_weights()
    # print(weights)
    for i, cell in enumerate(self.de_cells):
      s0, s1 = s1, cell(s0, s1, weights, mask, memory)

    logits = self.classifier(s1)
    logits = logits.view(-1, logits.size(-1))
    return logits

  def _loss(self, input, target, model):
    logits = model(input, target[:-1])
    return self._criterion(logits, target[1:].view(-1)) 

  def arch_parameters(self):
    # return self._arch_parameters
    # return list(self.alphas[0].parameters()) + list(self.alphas[1].parameters())
    return list(self.alphas_en.parameters()) + list(self.alphas_de.parameters())

  def parameters(self, recurse: bool = True, filter: bool = False):
    return_list = []
    for name, param in super().named_parameters(recurse=recurse):
      if ('alphas_en' in name or 'alphas_de' in name) and filter:
        pass
      else:
        return_list.append(param)

    return return_list

  def named_parameters(self, prefix: str = '', recurse: bool = True, filter: bool = False):
    return_list = []
    for name, param in super().named_parameters(prefix=prefix, recurse=recurse):
      if ('alphas_en' in name or 'alphas_de' in name) and filter:
        pass
      else:
        return_list.append((name, param))

    return return_list

  def genotype(self):

    def _parse(weights, decode=False):
      if decode:
        Layer_OPS = De_Layer_OPS
      else:
        Layer_OPS = En_Layer_OPS
      gene = []
      lookup = {'left_input': list(range(100)),
                # 'left_norm': list(Norm_OPS.keys()),
                'left_layer': list(Layer_OPS.keys()),
                'left_activation': list(Act_OPS.keys()),

                'right_input': list(range(100)),
                # 'right_norm': list(Norm_OPS.keys()),
                'right_layer': list(Layer_OPS.keys()),
                'right_activation': list(Act_OPS.keys()),

                'combine_func': list(Cmb_OPS.keys()),
                'norm': list(Norm_OPS.keys())}

      for block in weights:
        block_gene = {}
        for key in block:
          block_gene[key] = lookup[key][block[key].argmax().item()]
        gene.append(block_gene)

      return gene

    gene_encoder = _parse(self.alphas_en.return_weights(), False)
    gene_decoder = _parse(self.alphas_de.return_weights(), True)

    genotype = {'gene_encoder': gene_encoder, 'gene_decoder': gene_decoder}

    return genotype


