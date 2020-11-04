import torch
import torch.nn as nn
import torch.nn.functional as F

# OPS = {
#   'none' : lambda C, stride, affine: Zero(stride),
#   'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
#   'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
#   'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
#   'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
#   'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
#   'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
#   'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
#   'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
#   'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
#     nn.ReLU(inplace=False),
#     nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
#     nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
#     nn.BatchNorm2d(C, affine=affine)
#     ),
# }

Norm_OPS = {
  'layer_norm' : lambda C, affine: nn.LayerNorm(C, elementwise_affine=affine),
  'none' : lambda C, affine: Identity()
}

En_Layer_OPS = {
  'FFN_4' : lambda C, : FFN(C, 4),
  #'SA_h4' : lambda C, : En_SA(C, 4),
  'SA_h8' : lambda C, : En_SA(C, 8),
  #'SA_h16' : lambda C, : En_SA(C, 16),
  'conv1d_1x1' : lambda C, : Conv1d(C, C, 1, padding=0),
  'conv1d_3x3' : lambda C, : Conv1d(C, C, 3, padding=1),
  'sep_conv_3x3' : lambda C, : SepConv1d(C, C, 3, padding=1),
  'sep_conv_5x5' : lambda C, : SepConv1d(C, C, 5, padding=2),
  'sep_conv_7x7' : lambda C, : SepConv1d(C, C, 7, padding=3),
  #'sep_conv_9x9' : lambda C, : SepConv1d(C, C, 9, padding=4),
  #'sep_conv_11x11' : lambda C, : SepConv1d(C, C, 11, padding=5),
  'identity' : lambda C, : Identity(),
  'dead' : lambda C, : ZeroLayer(),
  'GLU' : lambda C, : GatedLinearUnit(C, C)
}

De_Layer_OPS = {
  'FFN_4' : lambda C, : FFN(C, 4),
  #'SA_h4' : lambda C, : De_SA(C, 4),
  'SA_h8' : lambda C, : De_SA(C, 8),
  #'SA_h16' : lambda C, : De_SA(C, 16),
  #'Att_En_h4' : lambda C, : De_At_En(C, 4),
  'Att_En_h8' : lambda C, : De_At_En(C, 8),
  #'Att_En_h16' : lambda C, : De_At_En(C, 16),
  'conv1d_1x1' : lambda C, : Conv1d(C, C, 1, padding=0, decode=True),
  'conv1d_3x3' : lambda C, : Conv1d(C, C, 3, padding=1, decode=True),
  'sep_conv_3x3' : lambda C, : SepConv1d(C, C, 3, padding=1, decode=True),
  'sep_conv_5x5' : lambda C, : SepConv1d(C, C, 5, padding=2, decode=True),
  'sep_conv_7x7' : lambda C, : SepConv1d(C, C, 7, padding=3, decode=True),
  #'sep_conv_9x9' : lambda C, : SepConv1d(C, C, 9, padding=4, decode=True),
  #'sep_conv_11x11' : lambda C, : SepConv1d(C, C, 11, padding=5, decode=True),
  'identity' : lambda C, : Identity(),
  'dead' : lambda C, : ZeroLayer(),
  'GLU' : lambda C, : GatedLinearUnit(C, C)
}

Act_OPS = {
  'Swish' : lambda : Swish(),
  'ReLU' : lambda : nn.ReLU(),
  'LeakyReLU' : lambda : nn.LeakyReLU(),
  'none' : lambda : Identity(),
}

Cmb_OPS = {
  'add' : lambda C: AddCombine(),
  'mul' : lambda C: MulCombine(),
  'cat' : lambda C: CatCombine(C),
}

class En_SA(nn.Module):

  def __init__(self, dim, head_num):
    super(En_SA, self).__init__()
    self.att = nn.MultiheadAttention(dim, head_num, dropout=0.1) 
    self.dropout = nn.Dropout(0.1)

  def forward(self, x, mask):
    return self.dropout(self.att(x, x, x, key_padding_mask=mask, need_weights=False)[0])

class De_SA(nn.Module):

  def __init__(self, dim, head_num):
    super(De_SA, self).__init__()
    self.att = nn.MultiheadAttention(dim, head_num, dropout=0.1) 
    self.dropout = nn.Dropout(0.1)

  def forward(self, x, mask, memory):
    device = x.device
    mask = torch.triu( torch.ones(len(x), len(x)) ).t().to(device) == 0
    return self.dropout(self.att(x, x, x, attn_mask=mask, need_weights=False)[0])

class De_At_En(nn.Module):

  def __init__(self, dim, head_num):
    super(De_At_En, self).__init__()
    self.att = nn.MultiheadAttention(dim, head_num, dropout=0.1) 
    self.dropout = nn.Dropout(0.1)

  def forward(self, x, mask, memory):
    return self.dropout(self.att(x, memory, memory, key_padding_mask=mask, need_weights=False)[0])

class FFN(nn.Module):

  def __init__(self, C_in, multiplier):
    super(FFN, self).__init__()
    self.op = nn.Sequential(
      nn.Linear(C_in, C_in*multiplier),
      nn.ReLU(),
      nn.Dropout(0.1),
      nn.Linear(C_in*multiplier, C_in),
      nn.Dropout(0.1),
    )

  def forward(self, x, mask, memory=None):
    return self.op(x)
  

class Conv1d(nn.Module):
  
  def __init__(self, C_in, C_out, kernel_size, padding, decode=False):
    super(Conv1d, self).__init__()
    self.conv = nn.Conv1d(C_in, C_out, kernel_size=kernel_size, stride=1, padding=0)
    assert padding == (kernel_size-1)//2
    self.padding = padding
    self.decode = decode
    self.dropout = nn.Dropout(0.1)

  def forward(self, x, mask, memory=None):
    device = x.device
    if self.decode:
      x = x.permute(1,2,0)
      x = torch.cat( (torch.zeros(x.size(0), x.size(1), self.padding*2, device=device), x), dim=-1)
    else:
      mask = mask.t().unsqueeze(-1).expand_as(x)
      x = x.masked_fill(mask, 0.0)
      x = x.permute(1,2,0)
      x = torch.cat( (torch.zeros(x.size(0), x.size(1), self.padding, device=device), x, torch.zeros(x.size(0), x.size(1), self.padding, device=device)), dim=-1)
    return self.dropout(self.conv(x).permute(2,0,1))


class SepConv1d(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, padding, decode=False):
    super(SepConv1d, self).__init__()
    self.op = nn.Sequential(
      nn.Conv1d(C_in, C_in, kernel_size, stride=1, padding=0, groups=C_in),
      nn.Conv1d(C_in, C_out, kernel_size=1, stride=1, padding=0),
      nn.Dropout(0.1),
    )
    assert padding == (kernel_size-1)//2
    self.padding = padding
    self.decode = decode
  
  def forward(self, x, mask, memory=None):
    device = x.device
    if self.decode:
      x = x.permute(1,2,0)
      x = torch.cat( (torch.zeros(x.size(0), x.size(1), self.padding*2, device=device), x), dim=-1)
    else:
      mask = mask.t().unsqueeze(-1).expand_as(x)
      x = x.masked_fill(mask, 0.0)
      x = x.permute(1,2,0)
      x = torch.cat( (torch.zeros(x.size(0), x.size(1), self.padding, device=device), x, torch.zeros(x.size(0), x.size(1), self.padding, device=device)), dim=-1)
    return self.op(x).permute(2,0,1)


class GatedLinearUnit(nn.Module):

  def __init__(self, C_in, C_out):
    super(GatedLinearUnit, self).__init__()
    self.Linear = nn.Linear(C_in, C_out)
    self.Gate = nn.Sequential(
      nn.Linear(C_in, C_out),
      nn.Sigmoid(),
    )
    self.dropout = nn.Dropout(0.1)

  def forward(self, x, mask, memory=None):
    return self.dropout(self.Linear(x) * self.Gate(x))


class ZeroLayer(nn.Module):

  def __init__(self):
    super(ZeroLayer, self).__init__()

  def forward(self, x, mask, memory=None):
    return x.mul(0.)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x, mask=None, memory=None):
    return x


class Swish(nn.Module):

  def __init__(self):
    super(Swish, self).__init__()
    self.beta = nn.Parameter(torch.ones(1,))

  def forward(self, x):
    return x * F.sigmoid( self.beta * x )


class AddCombine(nn.Module):

  def __init__(self):
    super(AddCombine, self).__init__()

  def forward(self, x, y):
    return x + y


class MulCombine(nn.Module):

  def __init__(self):
    super(MulCombine, self).__init__()

  def forward(self, x, y):
    return x * y


class CatCombine(nn.Module):

  def __init__(self, C):
    super(CatCombine, self).__init__()
    self.compress = nn.Linear(C*2, C)

  def forward(self, x, y):
    return self.compress( torch.cat((x,y), dim=-1) )


# class FReduce(nn.Module):

#   def __init__(self, C_in, C_out, affine=True):
#     super(FReduce, self).__init__()
#     assert C_out % 2 == 0
#     self.bn = nn.BatchNorm2d(C_in, affine=affine)
#     self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
#     self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
#     self.relu = nn.ReLU(inplace=False)

#   def forward(self, x):
#     x = self.bn(x.permute(0,3,1,2))
#     out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
#     out = self.relu(out)
#     return out.permute(0,2,3,1)


# class NReduce(nn.Module):

#   def __init__(self, C_in, C_out, affine=True):
#     super(NReduce, self).__init__()
#     self.op = nn.Sequential(
#       nn.BatchNorm2d(C_in, affine=affine),
#       nn.Conv2d(C_in, C_out, kernel_size=1, stride=1, padding=0, bias=False),
#       nn.ReLU(inplace=False),
#     )

#   def forward(self, x):
#     return self.op(x.permute(0,3,1,2)).permute(0,2,3,1)


# class ReLUConvBN(nn.Module):

#   def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
#     super(ReLUConvBN, self).__init__()
#     self.op = nn.Sequential(
#       nn.ReLU(inplace=False),
#       nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
#       nn.BatchNorm2d(C_out, affine=affine)
#     )

#   def forward(self, x):
#     return self.op(x)

# class DilConv(nn.Module):
    
#   def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
#     super(DilConv, self).__init__()
#     self.op = nn.Sequential(
#       nn.ReLU(inplace=False),
#       nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
#       nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#       nn.BatchNorm2d(C_out, affine=affine),
#       )

#   def forward(self, x):
#     return self.op(x)


# class SepConv(nn.Module):
    
#   def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
#     super(SepConv, self).__init__()
#     self.op = nn.Sequential(
#       nn.ReLU(inplace=False),
#       nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
#       nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
#       nn.BatchNorm2d(C_in, affine=affine),
#       nn.ReLU(inplace=False),
#       nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
#       nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#       nn.BatchNorm2d(C_out, affine=affine),
#       )

#   def forward(self, x):
#     return self.op(x)



# class Zero(nn.Module):

#   def __init__(self, stride):
#     super(Zero, self).__init__()
#     self.stride = stride

#   def forward(self, x):
#     if self.stride == 1:
#       return x.mul(0.)
#     return x[:,:,::self.stride,::self.stride].mul(0.)


# class FactorizedReduce(nn.Module):

#   def __init__(self, C_in, C_out, affine=True):
#     super(FactorizedReduce, self).__init__()
#     assert C_out % 2 == 0
#     self.relu = nn.ReLU(inplace=False)
#     self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
#     self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
#     self.bn = nn.BatchNorm2d(C_out, affine=affine)

#   def forward(self, x):
#     x = self.relu(x)
#     out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
#     out = self.bn(out)
#     return out

