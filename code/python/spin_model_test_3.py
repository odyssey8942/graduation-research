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
#sklearn.metrics.classification_report
from sklearn.metrics import classification_report,f1_score

# ハイパーパラメータ
DATA_NUM = 1
DATA_REPEAT = 10
EPOCH_NUM = 2400
REPEAT_NUM = EPOCH_NUM * 1
BATCH_SIZE = 10
DELAY_SIZE = 5

in_size = 10
hidden_size = 64
out_size = 1

    #正規化
def zscore(x):
    xmean = x.mean()
    xstd  = np.std(x)

    zscore = (x-xmean)/xstd
    return zscore

def arrangement_data(y):
    newy = float(y)
    newy = round(newy)
    if(newy >= 3.0):
        newy = 3.0
    elif(newy <= -3.0):
        newy = -3.0
    else:
        pass
    """
    if(newy == 3.0):
        return 1.0
    elif(newy == -3.0):
        return -1.0
    else:
        return 0.0
    #"""
    return (newy)

def pure_arrangement_data(y):
    newy = float(y)
    if(newy >= 3.0):
        newy = 3.0
    elif(newy <= -3.0):
        newy = -3.0
    else:
        pass
    return (newy)

class LSTM(Chain):
    def __init__(self, in_size, hidden_size, out_size):
        # クラスの初期化
        # :param in_size: 入力層のサイズ
        # :param hidden_size: 隠れ層のサイズ
        # :param out_size: 出力層のサイズ
        super(LSTM, self).__init__(
            xh = L.Linear(in_size, hidden_size),
            hh = L.LSTM(hidden_size, hidden_size),
            hy = L.Linear(hidden_size, out_size)
        )
 
    def __call__(self, x, t=None, train=False):
        # 順伝播の計算を行う関数
        # :param x: 入力値
        # :param t: 正解の予測値
        # :param train: 学習かどうか
        # :return: 計算した損失 or 予測値
        x = Variable(x)
        if train:
            t = Variable(t)
        h = self.xh(x)
        h = self.hh(h)
        y = self.hy(h)
        if train:
            return F.mean_squared_error(y, t)
        else:
            return y.data
 
    def reset(self):
        # 勾配の初期化とメモリの初期化
        self.cleargrads()
        self.hh.reset_state()





round = lambda x:(x*2+1)//2
# 予測

#roop = [50,100,150,200,250,300,350,400,450,500,550,600,650]
totalaccuracy = 0.0
print("\nPredict")
for i in range(1):
    predict = np.empty(0)
    purepredict = np.empty(0)
    test_x = []
    test_t = []

    test_path = "G:/CarData/train.csv" 
    model_path = "G:/cuda/test1/new_spin_model/10_64_1_5_10_4950_1000_use_train_spin_model.npz"
    model = LSTM(in_size=in_size, hidden_size=hidden_size, out_size=out_size)
    serializers.load_npz(model_path,model)
    csv_file = open(test_path, "r", encoding="utf_8", errors="", newline="\n" )
    f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)

    for row in f:
        """
        obj = row[2:8]
        del obj[3]
        obj[1] = float(obj[1])*float(10.0/36.0)
        test_x.append(obj)
        #"""
        #"""
        obj = row[1:12]
        del obj[4]
        obj[2] = float(obj[2])*float(10.0/36.0)
        test_x.append(obj)
        #"""
        """
        if(abs(float(row[12])) == 3.0):
            test_t.append(2.0)
        elif(abs(float(row[12])) == 2.0):
            test_t.append(1.0)
        elif(abs(float(row[12])) == 1.0 or abs(float(row[12])) == 0.0):
            test_t.append(0.0)
        else:
            pass
        #""
        if(float(row[12]) == 3.0):
            test_t.append(1.0)
        elif(float(row[12]) == -3.0):
            test_t.append(-1.0)
        else:
            test_t.append(0.0)
        #"""
        test_t.append((float(row[12])))
    del test_x[len(test_x) - DELAY_SIZE:]
    del test_t[0:DELAY_SIZE]
    test_x = np.array(test_x, dtype="float32")
    test_t = np.array(test_t, dtype="float32")

    test_x = zscore(test_x)

    for x in test_x:
        x=x.reshape(1,in_size)
        y = arrangement_data(model(x=x,train=False))
        #purey = pure_arrangement_data(model(x=x,train=False))
        predict = np.append(predict, y)
        #purepredict = np.append(purepredict, purey)
    #"""
    """
    target_names = ['class 1','class 0','class -1']
    print(f1_score(test_t,predict,average='weighted'))
    print(classification_report(test_t,predict,target_names=target_names))
    #"""
    """
    num = 0
    for i in range(len(predict)):
        if(predict[i] == test_t[i]):
            num+=1

    answer = num/len(predict)
    print(str(answer))

    totalaccuracy += answer
    #"""
    #"""
    N = len(test_t)
    plt.plot(range(N), test_t, color="red", label="t", linestyle="dashed")
    plt.plot(range(N), predict, color="blue", label="y")
    #plt.plot(range(N), purepredict, color="green", label="py")
    plt.legend(loc="upper left")
    plt.show()
    #"""

#print("accuracymean : " + str(totalaccuracy/float(roop)))