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
from sklearn.metrics import classification_report,f1_score

# ハイパーパラメータ
DATA_NUM = 1
DATA_REPEAT = 10
EPOCH_NUM = 2400
REPEAT_NUM = EPOCH_NUM * 1
BATCH_SIZE = 10
DELAY_SIZE = 5

in_size = 10
hidden_size = 128
out_size = 2

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
    if(abs(newy) == 3.0):
        #return 2.0
        return 1.0
    elif(abs(newy) == 2.0):
        #return 1.0
        return 0.0
    elif(abs(newy) == 1.0 or abs(newy) == 0.0):
        return 0.0
    else:
        pass
    #"""
    #"""
    if(newy == 3.0):
        return 1.0
    elif(newy == -3.0):
        return -1.0
    elif(newy <= 2.0 and newy >= -2.0):
        return 0.0
    else:
        pass
    #"""
    #return (newy)

def pure_arrangement_data(y):
    newy = float(y)
    """
    if(newy >= 3.0):
        newy = 3.0
    elif(newy <= -3.0):
        newy = -3.0
    else:
        pass
    #"""
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

roop = [650,680]
maxaccuracy = 0.0
maxnum = 0
print("\nPredict")
#for i in range(100):
for i in range(100):
    #purepredict = np.empty(0)
    #purey = np.empty(0)
    test_x = []
    test_t = []

    test_path = "G:/CarData/short_train.csv" 
    model_path = "G:/cuda/test1/new_spin_model/10_128_2_5_10_29900_" + str(i+1) + "0_use_train_spin_model.npz"
    #model_path = "G:/cuda/test1/new_spin_model/10_128_2_5_10_4950_1000_use_train_spin_model.npz"
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
            #test_t.append(2.0)
            test_t.append(1.0)
        elif(abs(float(row[12])) == 2.0):
            #test_t.append(1.0)
            test_t.append(0.0)
        elif(abs(float(row[12])) == 1.0 or abs(float(row[12])) == 0.0):
            test_t.append(0.0)
        else:
            pass
        #"""
        """
        if(float(row[12]) == 3.0):
            test_t.append(1.0)
        elif(float(row[12]) == -3.0):
            test_t.append(-1.0)
        elif(float(row[12]) <= 2.0 and float(row[12]) >= -2.0):
            test_t.append(0.0)
        else:
            pass
        #"""
        test_t.append(((row[6:8])))
    del test_x[len(test_x) - DELAY_SIZE:]
    del test_t[0:DELAY_SIZE]
    test_x = np.array(test_x, dtype="float32")
    test_t = np.array(test_t, dtype="float32")

    test_x = zscore(test_x)
    purepredict = np.empty(0)
    purey = np.empty(0)
    name=["class 3","class 2","class 1","class 0","class -1","class -2","class -3"]
    #name=["class 1","class 0"]
    for x in test_x:
        x=x.reshape(1,in_size)
        #y = arrangement_data(model(x=x,train=False))
        purey = model(x=x,train=False)
        #predict = np.append(predict, y)
        purepredict = np.append(purepredict, purey)
    """
    print((i+1)*100)
    print(f1_score(test_t,predict,average="macro"))
    print(classification_report(test_t,predict,target_names=name))
    #"""
    #"""
    N = len(test_t)
    purepredict = np.reshape(purepredict,(len(purepredict)//out_size,out_size))
    #print(test_t)
    #print(purepredict)
    test_bf = []
    test_br = []
    predict_bf = []
    predict_br = []

    p = []
    t = []
    
    for row in test_t:
        """
        if(float(row[0]) > 10 or float(row[1]) > 10):
            if(row[0] >= row[1]):
                t.append(3)
            elif(row[0] < row[1]):
                t.append(-3)
        elif(float(row[0]) > 4 or float(row[1]) > 4):
            if(row[0] >= row[1]):
                t.append(2)
            elif(row[0] < row[1]):
                t.append(-2)
        elif(float(row[0]) > 0 or float(row[1]) > 0):
            if(row[0] >= row[1]):
                t.append(1)
            elif(row[0] < row[1]):
                t.append(-1)
        else:
            t.append(0)
        #"""
        if(float(row[0]) > 10 or float(row[1]) > 10):
            if(row[0] >= row[1]):
                t.append(1)
            elif(row[0] < row[1]):
                t.append(-1)
        else:
            t.append(0)
        """
        test_bf.append(float(row[0]))
        test_br.append(float(row[1]))
        #"""
    for row in purepredict:
        """
        if(float(row[0]) > 10 or float(row[1]) > 10):
            if(row[0] >= row[1]):
                p.append(3)
            elif(row[0] < row[1]):
                p.append(-3)
        elif(float(row[0]) > 4 or float(row[1]) > 4):
            if(row[0] >= row[1]):
                p.append(2)
            elif(row[0] < row[1]):
                p.append(-2)
        elif(float(row[0]) > 0 or float(row[1]) > 0):
            if(row[0] >= row[1]):
                p.append(1)
            elif(row[0] < row[1]):
                p.append(-1)
        else:
            p.append(0)
        #"""
        if(float(row[0]) > 10 or float(row[1]) > 10):
            if(row[0] >= row[1]):
                p.append(1)
            elif(row[0] < row[1]):
                p.append(-1)
        else:
            p.append(0)
        """
        predict_bf.append(float(row[0]))
        predict_br.append(float(row[1]))
        #"""
    """
    path_w = "plot.csv"
    f = open(path_w, mode="w")
    for i in range(len(predict_bf)):
        f.write(str(test_bf[i]) + "," + str(predict_bf[i]) + "\n")
    #"""
    #print((i+1)*10)
    print(f1_score(t,p,average="weighted"))
    #"""
    #print(classification_report(t,p,target_names=name))
    if(f1_score(t,p,average="weighted") > maxaccuracy):
        maxaccuracy = f1_score(t,p,average="weighted")
        maxnum = (i+1)*10
    #"""
    """
    plt.plot(range(N), test_bf, color="red", label="test_bf")
    plt.plot(range(N), test_br, color="blue", label="test_br")
    plt.plot(range(N), predict_bf, color="green", label="predict_bf")
    plt.plot(range(N), predict_br, color="yellow", label="predict_br")
    plt.show()
    #"""
    """
    plt.plot(range(N), t, color="red", label="t")
    plt.plot(range(N), p, color="blue", label="p")
    plt.legend(loc="upper left")
    plt.show()
    #"""
print(maxaccuracy,maxnum)