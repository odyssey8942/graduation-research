#coding: UTF-8
import math
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

class LSTM(Chain):
    def __init__(self, in_size, hidden_size, out_size):
        # クラスの初期化
        # :param in_size: 入力層のサイズ
        # :param hidden_size: 隠れ層のサイズ
        # :param out_size: 出力層のサイズ
        super(LSTM, self).__init__(
            xh = L.Linear(in_size, hidden_size),
            hh = L.LSTM(hidden_size, hidden_size),
            hy =  L.Linear(hidden_size, out_size)
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

 
# 学習
 
EPOCH_NUM = 5000
HIDDEN_SIZE = 32
BATCH_ROW_SIZE = 100 # 分割した時系列をいくつミニバッチに取り込むか
BATCH_COL_SIZE = 100 # ミニバッチで分割する時系列数

def get_data(x,t,i):
    #教師データ
    train_x, train_t = [], []
    path = "C:/Users/odyssey8942/Documents/get_car_data/TeacherData/ (" +str(i+1)+ ").csv"    
    csv_file = open(path, "r", encoding="utf_8", errors="", newline="\n" )
    f = csv.reader(csv_file, delimiter=" ", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)

    #教師データ変換
    for row in f:
        train_x.append(row[6:8])
        train_t.append(row[14:15])

    del train_x[len(train_x)-10:]
    del train_x[:len(train_x)-20]
    del train_t[:len(train_t)-20]

    train_x = np.array(train_x, dtype="float32")
    train_t = np.array(train_t, dtype="float32")

    return train_x,train_t

in_size = 2
out_size = 1

# モデルの定義
model = LSTM(in_size=in_size, hidden_size=HIDDEN_SIZE, out_size=out_size)
optimizer = optimizers.Adam()
optimizer.setup(model)

# 学習開始
print("Train")
st = dt.datetime.now()
for epoch in range(EPOCH_NUM):
    x,t = [],[]
    x,t = get_data(x,t,epoch%10)
    loss = 0
    total_loss = 0
    model.reset() # 勾配とメモリの初期化
    print(epoch%10)
    for i in range(len(x)):
        if(math.isnan(x[i][0])):
            x[i][0] = 0.0
        if(math.isnan(x[i][1])):
            x[i][1] = 0.0
        #print(x[i][0],x[i][1])
        #print(i,x[i])
        loss += model(x=np.array(x[i],dtype="float32").reshape(1,2), t=np.array(t[i],dtype="float32").reshape(1,1), train=True)
    loss.backward()
    loss.unchain_backward()
    total_loss += loss.data
    optimizer.update()
    if (epoch+1) % 100 == 0:
        ed = dt.datetime.now()
        print("epoch:\t{}\ttotal loss:\t{}\ttime:\t{}".format(epoch+1, total_loss, ed-st))
        st = dt.datetime.now()

# 予測

print("\nPredict")
predict = np.empty(0)
test_x = []
test_t = []
test_path = "C:/Users/odyssey8942/Documents/get_car_data/TeacherData/ (4990).csv"

csv_file = open(test_path, "r", encoding="utf_8", errors="", newline="\n" )
f = csv.reader(csv_file, delimiter=" ", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
flag = 0

for row in f:
    test_x.append(row[6:8])
    test_t.append(row[14:15])
del test_x[0:10]
del test_t[0:10]
test_x = np.array(test_x, dtype="float32")
test_t = np.array(test_t, dtype="float32")



for x in test_x:
    x=x.reshape(1,2)
    if(math.isnan(x[0][0])):
        x[0][0] = 0
    if(math.isnan(x[0][1])):
        x[0][1] = 0
    y = model(x=x,train=False)
    predict = np.append(predict, y)


N = len(test_t)
plt.plot(range(N), test_t, color="red", label="t")
plt.plot(range(N), predict, color="blue", label="y")
plt.legend(loc="upper left")
plt.show()
"""
# 予測
 
print("\nPredict")
predict = np.empty(0) # 予測時系列
inseq_size = 50
inseq = train_data[:inseq_size] # 予測直前までの時系列
for _ in range(N - inseq_size):
    model.reset() # メモリを初期化
    for i in inseq: # モデルに予測直前までの時系列を読み込ませる
        x = np.array([[i]], dtype="float32")
        print(x.shape)
        y = model(x=x, train=False)
    predict = np.append(predict, y) # 最後の予測値を記録
    # モデルに読み込ませる予測直前時系列を予測値で更新する
    inseq = np.delete(inseq, 0)
    inseq = np.append(inseq, y)
 
plt.plot(range(N+1), train_data, color="red", label="t")
plt.plot(range(inseq_size+1, N+1), predict, color="blue", label="y")
plt.legend(loc="upper left")
plt.show()
#"""