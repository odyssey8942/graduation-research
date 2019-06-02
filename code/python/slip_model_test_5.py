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
DATA_NUM = 24
DATA_REPEAT = 100
EPOCH_NUM = 2480
REPEAT_NUM = EPOCH_NUM * 1
BATCH_SIZE = 10
DELAY_SIZE = 5

in_size = 10
hidden_size = 128
out_size = 4

    #正規化
def zscore(x):
    xmean = x.mean()
    xstd  = np.std(x)

    zscore = (x-xmean)/xstd
    return zscore

def arrangement_data(y):
    newy = float(y)
    #"""
    newy = round(newy)
    if(newy >= 1.0):
        newy = 1.0
    elif(newy <= 0.0):
        newy = 0.0
    else:
        pass
    #"""
    return newy

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
roop = [700,750]
maxaccuracy = 0.0
maxnum = 0.0
print("\nPredict")
for i in range(100):
    predict = np.empty(0)
    isslip_train = []
    isslip_predict = []
    test_x = []
    test_t = []
    test_path = "G:/CarData/short_train.csv" 
    model_path = "G:/cuda/test1/new_slip_model/10_128_4_5_10_29900_" + str(i+1) + "0_use_train_slip_model.npz"
    #model_path = "H:/gr/cuda/test1/new_slip_model/10_128_4_5_10_4950_60_use_train_slip_model.npz"

    model = LSTM(in_size=in_size, hidden_size=hidden_size, out_size=out_size)
    serializers.load_npz(model_path,model)
    csv_file = open(test_path, "r", encoding="utf_8", errors="", newline="\n" )
    f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)

    for row in f:
        """
        obj = row[3:12]
        del obj[2]
        obj[0] = float(obj[0])*float(10.0/36.0)
        test_x.append(obj)
        #"""
        #"""
        obj = row[1:12]
        del obj[4]
        obj[2] = float(obj[2])*float(10.0/36.0)
        test_x.append(obj)
        #"""
        test_t.append(row[8:12])

    del test_x[len(test_x) - DELAY_SIZE:]
    del test_t[0:DELAY_SIZE]
    test_x = np.array(test_x, dtype="float32")
    test_t = np.array(test_t, dtype="float32")

    test_x_zscore = zscore(test_x)

    predict = np.empty(0)
    y = np.empty(0)

    for x in test_x_zscore:
        x=x.reshape(1,in_size)
        y = model(x=x,train=False)
        predict = np.append(predict, y)
    """
    name=["class 1","class 0"]
    print((i+1)*10)
    print(f1_score(test_t,predict,average="macro"))
    print(classification_report(test_t,predict,target_names=name))
    #"""
    predict = np.reshape(predict,(len(predict)//out_size,out_size))
    test_fl = []
    test_fr = []
    test_rl = []
    test_rr = []
    predict_fl = []
    predict_fr = []
    predict_rl = []
    predict_rr = []
    for row in test_t:
        """
        test_fl.append(row[0])
        test_fr.append(row[1])
        test_rl.append(row[2])
        test_rr.append(row[3])
        #"""    
        if(row[0] > 0.2 or row[1] > 0.2 or row[2] > 0.2 or row[3] > 0.2):
            isslip_train.append(1)
        else:
            isslip_train.append(0)
    #print(predict.shape)
    for row in predict:
        """
        predict_fl.append(row[0])
        predict_fr.append(row[1])
        predict_rl.append(row[2])
        predict_rr.append(row[3])
        #"""
        if(row[0] > 0.2 or row[1] > 0.2 or row[2] > 0.2 or row[3] > 0.2):
            isslip_predict.append(1)
        else:
            isslip_predict.append(0)

    print(f1_score(isslip_train,isslip_predict,average="weighted"))
    if(f1_score(isslip_train,isslip_predict,average="weighted") > maxaccuracy):
        maxaccuracy = f1_score(isslip_train,isslip_predict,average="weighted")
        maxnum = (i+1)*10
    #N = len(test_t)
    """
    plt.plot(range(N), test_fl, color="red", label="test_fl")
    #plt.plot(range(N), test_fr, color="blue", label="test_fr")
    #plt.plot(range(N), test_rl, color="green", label="test_rl")
    #plt.plot(range(N), test_rr, color="yellow", label="test_rr")

    plt.plot(range(N), predict_fl, color="orange", label="predict_fl")
    #plt.plot(range(N), predict_fr, color="aqua", label="predict_fr")
    #plt.plot(range(N), predict_rl, color="lime", label="predict_rl")
    #plt.plot(range(N), predict_rr, color="gold", label="predict_rr")
    plt.legend(loc="upper left")
    plt.show()
    #"""

print(maxaccuracy,maxnum)