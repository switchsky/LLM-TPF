from data_provider.data_factory import data_provider
from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.m4_summary import M4Summary
from utils.cmLoss import cmLoss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas
file_path = './m4_results/' + 'TPF' + '/'+'Weekly_forecast.csv'
m4_summary = M4Summary(file_path,'./datasets/m4')
# m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
smape_results, owa_results, mape, mase = m4_summary.evaluate()
print('smape:', smape_results)
print('mape:', mape)
print('mase:', mase)
print('owa:', owa_results)