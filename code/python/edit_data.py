#coding: UTF-8
import math
import random
import pandas as pd
import datetime as dt
import numpy as np
import cupy as cp
import matplotlib.pylab as plt
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, optimizer, serializers, utils, Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import csv

predict,train = [],[]

path = "G:/gr/cuda/test1/PredictData/Spin/5_64_1_360000_predict.txt"
round = lambda x:(x*2+1)//2
data = open(path,"r")

for line in data:
    newline = float(line)
    newline = round(newline)
    if(newline >= 3.0):
        newline = 3.0
    elif(newline <= -3.0):
        newline = -3.0
    else:
        pass
    predict.append(abs(newline))
data.close()

path = "G:/gr/cuda/test1/PredictData/Spin/5_64_1_360000_train.txt"
data = open(path,"r")
for line in data:
    newline = float(line)
    train.append(abs(newline))
data.close()

num = 0
for i in range(len(predict)):
    if(predict[i] == train[i]):
        num+=1

answer = num/len(predict)
print(answer)

N = len(predict)
plt.plot(range(N), train, color="red", label="t")
plt.plot(range(N), predict, color="blue", label="y")
plt.legend(loc="upper left")
plt.show()
