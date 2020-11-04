import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import numpy as np

base_path = './eval-EXP-20200928-131358/'
loss_path = base_path + 'loss'
fig_path = base_path + 'curve_fig.png'

f = open(loss_path, 'rb')
train_loss, valid_loss = pickle.load(f)
f.close()

x = np.arange(len(train_loss)) + 1
plt.plot(x, train_loss, label='train loss')
plt.plot(x, valid_loss, label='valid loss')
plt.legend()
plt.savefig(fig_path)
plt.cla()