import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


# class MixedOp(nn.Module):

#   def __init__(self, C, stride):
#     super(MixedOp, self).__init__()
#     self._ops = nn.ModuleList()
#     for primitive in PRIMITIVES:
#       op = OPS[primitive](C, stride, False)
#       if 'pool' in primitive:
#         op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
#       self._ops.append(op)

#   def forward(self, x, weights):
#     return sum(w * op(x) for w, op in zip(weights, self._ops))

class TwoBranchOp(nn.Module):

  def __init__(self, Width, C):
    super(TwoBranchOp, self).__init__()
    self._norm_ops_l = nn.ModuleList()
    self._norm_ops_r = nn.ModuleList()
    for key in Norm_OPS:
      self._norm_ops_l.append(Norm_OPS[key](C, False))
      self._norm_ops_r.append(Norm_OPS[key](C, False))

    self._layer_ops_l = nn.ModuleList()
    self._layer_ops_r = nn.ModuleList()
    for key in Layer_OPS:
      self._layer_ops_l.append(Layer_OPS[key](C, Width))
      self._layer_ops_r.append(Layer_OPS[key](C, Width))

    self._act_ops_l = nn.ModuleList()
    self._act_ops_r = nn.ModuleList()
    for key in Act_OPS:
      self._act_ops_l.append(Act_OPS[key]())
      self._act_ops_r.append(Act_OPS[key]())

    self._cmb_ops = nn.ModuleList()
    for key in Cmb_OPS:
      self._cmb_ops.append(Cmb_OPS[key](C))

  def forward(self, x, weights):  #weights: dict {left_input:[], left_norm:[], left_layer:[], left_activation:[], right.... combine_func:[]}
    x=torch.cat( list(map(lambda x: x.unsqueeze(0), x)) ,dim=0)
    left_x=torch.sum( x * weights['left_input'].view(-1,1,1,1,1) , dim=0)
    left_x=sum(w * op(left_x) for w, op in zip(weights['left_norm'], self._norm_ops_l))
    left_x=sum(w * op(left_x) for w, op in zip(weights['left_layer'], self._layer_ops_l))
    left_x=sum(w * op(left_x) for w, op in zip(weights['left_activation'], self._act_ops_l))

    right_x=torch.sum( x * weights['right_input'].view(-1,1,1,1,1) , dim=0)
    right_x=sum(w * op(right_x) for w, op in zip(weights['right_norm'], self._norm_ops_r))
    right_x=sum(w * op(right_x) for w, op in zip(weights['right_layer'], self._layer_ops_r))
    right_x=sum(w * op(right_x) for w, op in zip(weights['right_activation'], self._act_ops_r))

    x=sum(w * op(left_x, right_x) for w, op in zip(weights['combine_func'], self._cmb_ops))

    return x
    

class EvolvedCell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, width):
    super(EvolvedCell, self).__init__()
    self.reduction = reduction

    preprocess0_ops = []
    if reduction_prev:
      preprocess0_ops.append( FReduce(C_prev_prev, C_prev_prev, affine=False) )

    if reduction:
      preprocess0_ops.append( FReduce(C_prev_prev, C, affine=False) )
      self.preprocess1 = FReduce(C_prev, C, affine=False)
    else:
      preprocess0_ops.append( NReduce(C_prev_prev, C, affine=False) )
      self.preprocess1 = NReduce(C_prev, C, affine=False)
    self.preprocess0 = nn.Sequential(*preprocess0_ops)
    
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    for i in range(self._steps):
      op = TwoBranchOp(width, C)
      self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = self._ops[i](states, weights[i])
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=3)


# class Cell(nn.Module):

#   def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
#     super(Cell, self).__init__()
#     self.reduction = reduction

#     if reduction_prev:
#       self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
#     else:
#       self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
#     self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
#     self._steps = steps
#     self._multiplier = multiplier

#     self._ops = nn.ModuleList()
#     self._bns = nn.ModuleList()
#     for i in range(self._steps):
#       for j in range(2+i):
#         stride = 2 if reduction and j < 2 else 1
#         op = MixedOp(C, stride)
#         self._ops.append(op)

#   def forward(self, s0, s1, weights):
#     s0 = self.preprocess0(s0)
#     s1 = self.preprocess1(s1)

#     states = [s0, s1]
#     offset = 0
#     for i in range(self._steps):
#       s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
#       offset += len(states)
#       states.append(s)

#     return torch.cat(states[-self._multiplier:], dim=1)

class BlockWeights(nn.Module):
  def __init__(self, input_num):
    super(BlockWeights, self).__init__()
    self.param = nn.ParameterDict({'left_input': nn.Parameter( 1e-3*torch.randn(input_num,) ),
                                   'left_norm': nn.Parameter( 1e-3*torch.randn(len(Norm_OPS),) ),
                                   'left_layer': nn.Parameter( 1e-3*torch.randn(len(Layer_OPS),) ),
                                   'left_activation': nn.Parameter( 1e-3*torch.randn(len(Act_OPS),) ),

                                   'right_input': nn.Parameter( 1e-3*torch.randn(input_num,) ),
                                   'right_norm': nn.Parameter( 1e-3*torch.randn(len(Norm_OPS),) ),
                                   'right_layer': nn.Parameter( 1e-3*torch.randn(len(Layer_OPS),) ),
                                   'right_activation': nn.Parameter( 1e-3*torch.randn(len(Act_OPS),) ),

                                   'combine_func': nn.Parameter( 1e-3*torch.randn(len(Cmb_OPS),) ),
                                    })

  def return_weights(self,):
    reg_weights={}
    for key in self.param:
      reg_weights[key]=F.softmax( self.param[key], dim=0)
    return reg_weights


class CellWeights(nn.Module):
  def __init__(self, steps):
    super(CellWeights, self).__init__()
    self.param = nn.ModuleList()
    for i in range(steps):
      self.param.append(BlockWeights(i+2))

  def return_weights(self):
    reg_weights=[]
    for block in self.param:
      reg_weights.append( block.return_weights() )
    return reg_weights

class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, width, steps=4, multiplier=3, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self._width = width

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    width_curr = width
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        width_curr = width_curr // 2
        assert width_curr % 2 == 0
        reduction = True
      else:
        reduction = False
      cell = EvolvedCell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, width_curr)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self.alphas = [CellWeights(steps).cuda(), CellWeights(steps).cuda()]
    # self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion, self._width).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = s1 = self.stem(input).permute(0,2,3,1)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = self.alphas[1].return_weights()
      else:
        weights = self.alphas[0].return_weights()
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1.permute(0,3,1,2))
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  # def _initialize_alphas(self):
  #   k = sum(1 for i in range(self._steps) for n in range(2+i))
  #   num_ops = len(PRIMITIVES)

  #   self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
  #   self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
  #   self._arch_parameters = [
  #     self.alphas_normal,
  #     self.alphas_reduce,
  #   ]

  def arch_parameters(self):
    # return self._arch_parameters
    return list(self.alphas[0].parameters()) + list(self.alphas[1].parameters())

  def genotype(self):

    def _parse(weights):
      gene = []
      lookup = {'left_input': list(range(100)),
                'left_norm': list(Norm_OPS.keys()),
                'left_layer': list(Layer_OPS.keys()),
                'left_activation': list(Act_OPS.keys()),

                'right_input': list(range(100)),
                'right_norm': list(Norm_OPS.keys()),
                'right_layer': list(Layer_OPS.keys()),
                'right_activation': list(Act_OPS.keys()),

                'combine_func': list(Cmb_OPS.keys())}

      for block in weights:
        block_gene = {}
        for key in block:
          block_gene[key] = lookup[key][block[key].argmax().item()]
        gene.append(block_gene)

      return gene

    gene_normal = _parse(self.alphas[0].return_weights())
    gene_reduce = _parse(self.alphas[1].return_weights())

    genotype = {'gene_normal': gene_normal, 'gene_reduce': gene_reduce}

    return genotype


  # def genotype(self):

  #   def _parse(weights):
  #     gene = []
  #     n = 2
  #     start = 0
  #     for i in range(self._steps):
  #       end = start + n
  #       W = weights[start:end].copy()
  #       edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
  #       for j in edges:
  #         k_best = None
  #         for k in range(len(W[j])):
  #           if k != PRIMITIVES.index('none'):
  #             if k_best is None or W[j][k] > W[j][k_best]:
  #               k_best = k
  #         gene.append((PRIMITIVES[k_best], j))
  #       start = end
  #       n += 1
  #     return gene

  #   gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
  #   gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

  #   concat = range(2+self._steps-self._multiplier, self._steps+2)
  #   genotype = Genotype(
  #     normal=gene_normal, normal_concat=concat,
  #     reduce=gene_reduce, reduce_concat=concat
  #   )
  #   return genotype

