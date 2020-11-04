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
  'batch_norm' : lambda C, affine: BatchNorm2d(C, affine=affine),
  'none' : lambda C, affine: Identity()
}

Layer_OPS = {
  'FFN_4' : lambda C, W: FFN(C, 4),
  'SA_h4' : lambda C, W: SelfAttention(W, C, 4),
  'SA_h8' : lambda C, W: SelfAttention(W, C, 8),
  'conv2d_3x3' : lambda C, W: Conv2d(C, C, 3, padding=1),
  'conv2d_5x5' : lambda C, W: Conv2d(C, C, 5, padding=2),
  'sep_conv_3x3' : lambda C, W: SepConv2d(C, C, 3, padding=1),
  'sep_conv_5x5' : lambda C, W: SepConv2d(C, C, 5, padding=2),
  'sep_conv_7x7' : lambda C, W: SepConv2d(C, C, 7, padding=3),
  'sep_conv_9x9' : lambda C, W: SepConv2d(C, C, 9, padding=4),
  'sep_conv_11x11' : lambda C, W: SepConv2d(C, C, 11, padding=5),
  'identity' : lambda C, W: Identity(),
  'dead' : lambda C, W: ZeroLayer(),
  'GLU' : lambda C, W: GatedLinearUnit(C, C)
  #'conv2d_1x1' : lambda C, W: nn.Conv2d(C, C, 1, 1, padding=0, bias=True),
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

class BatchNorm2d(nn.Module):
  
  def __init__(self, C, affine):
    super(BatchNorm2d, self).__init__()
    self.bn = nn.BatchNorm2d(C, affine=affine)

  def forward(self, x):
    return self.bn(x.permute(0,3,1,2)).permute(0,2,3,1)

class FFN(nn.Module):

  def __init__(self, C_in, multiplier):
    super(FFN, self).__init__()
    self.op = nn.Sequential(
      nn.Linear(C_in, C_in*multiplier),
      nn.ReLU(),
      nn.Linear(C_in*multiplier, C_in)
    )

  def forward(self, x):
    return self.op(x)
  

class Conv2d(nn.Module):
  
  def __init__(self, C_in, C_out, kernel_size, padding):
    super(Conv2d, self).__init__()
    self.conv = nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=1, padding=padding)

  def forward(self, x):
    return self.conv(x.permute(0,3,1,2)).permute(0,2,3,1)


class SepConv2d(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, padding):
    super(SepConv2d, self).__init__()
    self.op = nn.Sequential(
      nn.Conv2d(C_in, C_in, kernel_size, stride=1, padding=padding, groups=C_in),
      nn.Conv2d(C_in, C_out, kernel_size=1, stride=1, padding=0),
    )
  
  def forward(self, x):
    return self.op(x.permute(0,3,1,2)).permute(0,2,3,1)


class GatedLinearUnit(nn.Module):

  def __init__(self, C_in, C_out):
    super(GatedLinearUnit, self).__init__()
    self.Linear = nn.Linear(C_in, C_out)
    self.Gate = nn.Sequential(
      nn.Linear(C_in, C_out),
      nn.Sigmoid(),
    )

  def forward(self, x):
    return self.Linear(x) * self.Gate(x)


class ZeroLayer(nn.Module):

  def __init__(self):
    super(ZeroLayer, self).__init__()

  def forward(self, x):
    return x.mul(0.)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class SelfAttention(nn.Module):

  def __init__(self, width, depth, head_num):
    super(SelfAttention, self).__init__() #we set D_h = D_in = D_out
    height = width
    if depth%2 != 0:
      D_h = depth-1
    else:
      D_h = depth
    self.D_h = D_h
    self.head_num = head_num
    self.K_w = nn.Linear(depth, D_h*head_num)
    self.Q_w = nn.Linear(depth, D_h*head_num)
    self.V_w = nn.Linear(depth, D_h*head_num)
    self.proj = nn.Linear(D_h*head_num, depth)
    # self.row_emb = None
    # self.col_emb = None

    self.row_emb = nn.Embedding(height*2-1, self.D_h//2)
    self.register_buffer('row_mat', (torch.arange(height).unsqueeze(0) - torch.arange(height).unsqueeze(1))+height-1)

    self.col_emb = nn.Embedding(width*2-1, self.D_h//2)
    self.register_buffer('col_mat', (torch.arange(width).unsqueeze(1) - torch.arange(width).unsqueeze(0))+width-1)

  def forward(self, x):
    batch_size, height, width, depth = x.size()
    # device=self.K_w.weight.device
    # if self.row_emb is None:
    #   self.row_emb = nn.Embedding(height*2-1, self.D_h//2).to(device)
    #   self.register_buffer('row_mat', (torch.arange(height).to(device).unsqueeze(0) - torch.arange(height).to(device).unsqueeze(1))+height-1)
    # if self.col_emb is None:
    #   self.col_emb = nn.Embedding(width*2-1, self.D_h//2).to(device)
    #   self.register_buffer('col_mat', (torch.arange(width).to(device).unsqueeze(1) - torch.arange(width).to(device).unsqueeze(0))+width-1)
    K = self.K_w(x).view(batch_size, height, width, self.head_num, self.D_h)
    Q = self.Q_w(x).view(batch_size, height, width, self.head_num, self.D_h)
    V = self.V_w(x).view(batch_size, height, width, self.head_num, self.D_h)
    attention_score = torch.einsum('bijhd,bklhd->bijhkl', Q, K)

    row_embbedding = self.row_emb(self.row_mat)
    row_score = torch.einsum('bijhd,ikd->bijhk', Q[:,:,:,:,:self.D_h//2], row_embbedding).unsqueeze(-1)

    col_embbedding = self.col_emb(self.col_mat)
    col_score = torch.einsum('bijhd,jkd->bijhk', Q[:,:,:,:,self.D_h//2:], col_embbedding).unsqueeze(-2)

    attention_score = attention_score + (row_score + col_score)
    attention_score = F.softmax(attention_score.view(batch_size, height, width, self.head_num, -1), dim=-1).view(batch_size, height, width, self.head_num, height, width)

    attended = torch.einsum('bijhkl,bklhd->bijhd', attention_score, V).reshape(batch_size, height, width, -1)

    return self.proj(attended)


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


class FReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FReduce, self).__init__()
    assert C_out % 2 == 0
    self.bn = nn.BatchNorm2d(C_in, affine=affine)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.relu = nn.ReLU(inplace=False)

  def forward(self, x):
    x = self.bn(x.permute(0,3,1,2))
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.relu(out)
    return out.permute(0,2,3,1)


class NReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(NReduce, self).__init__()
    self.op = nn.Sequential(
      nn.BatchNorm2d(C_in, affine=affine),
      nn.Conv2d(C_in, C_out, kernel_size=1, stride=1, padding=0, bias=False),
      nn.ReLU(inplace=False),
    )

  def forward(self, x):
    return self.op(x.permute(0,3,1,2)).permute(0,2,3,1)


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

