from model_search import Network

class Cup:

    #构造函数，初始化属性值
    def __init__(self,capacity,color):
        self.capacity=capacity
        self.color=color

    def retain_water(self, x):
        return x + 'aaa'
        # print("杯子颜色："+self.color+"，杯子容量："+self.capacity+",正在装水.")

    def keep_warm(self):
        print("杯子颜色："+self.color+"，杯子容量："+self.capacity+",正在保温.")

# class Luminous_Cup(Cup):

#     #构造函数，调用父类的构造函数初始化属性值
#     def __init__(self,capacity,color):
#         super().__init__(capacity,color)

#     #方法重写
#     def retain_water(self, x):
#         x = x + 'ccc' + super().retain_water(x)
#         # print("杯子颜色："+self.color+"，杯子容量："+self.capacity+",正在装水,正在发光...")
#         return x

#     def glow(self):
#         print("我正在发光...")


# # currentCup=Luminous_Cup('300ml','翠绿色')
# #调用子类中的retain_water()方法
# # print(currentCup.retain_water('qqq'))
# #调用父类中的retain_water()方法
# # super(Luminous_Cup,currentCup).retain_water()

# net = Network(64, 10, 10, 2, 2, 'criterion', 0)
# print(len(list(super(Network, net).parameters())))
# # print(len(net.arch_parameters()))
# # print(len(net.parameters()))
# # print(list(map(lambda x: x[0], super(Network, net).named_parameters())))
import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.params = nn.ParameterDict({
                'left': nn.Parameter(torch.randn(5, 10)),
                'right': nn.Parameter(torch.randn(5, 10))
        })
        self.params2 = nn.Linear(3,4)

    def forward(self, x, choice):
        x = self.params[choice].mm(x)
        return x

m = MyModule()
m = nn.DataParallel(m.cuda())
# print(list(m.parameters()))
for module_prefix, module in list(m.named_modules()):
    print(module._parameters)