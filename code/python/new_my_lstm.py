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
    
    #正規化
def zscore(x):
    xmean = x.mean()
    xstd  = np.std(x)

    zscore = (x-xmean)/xstd
    return zscore

def get_data(x,t):
    #教師データ
    train_x, train_t = [], []
    path = "C:/Users/odyssey8942/Documents/new_get_car_data/TeacherData3/control_myself_cardata.csv"    
    csv_file = open(path, "r", encoding="utf_8", errors="", newline="\n" )
    f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)

    #教師データ変換
    for row in f:
        obj = row[2:8]
        del obj[3]
        train_x.append(obj)
        train_t.append(row[13:14])

    del train_x[len(train_x)-10:]
    del train_t[0:10]

    train_x = np.array(train_x, dtype="float32")
    train_t = np.array(train_t, dtype="float32")

    train_x_zscore = zscore(train_x)

    print(train_x)
    print(train_x_zscore)

    return train_x_zscore,train_t

# 学習
EPOCH_NUM = 4500
REPEAT_NUM = EPOCH_NUM * 80
HIDDEN_SIZE = 64
BATCH_SIZE = 20

in_size = 5
out_size = 1

index = random.sample(range(4969), EPOCH_NUM)
# モデルの定義
model = LSTM(in_size=in_size, hidden_size=HIDDEN_SIZE, out_size=out_size)
optimizer = optimizers.Adam()
optimizer.setup(model)

# 学習開始
print("Train")
st = dt.datetime.now()
x,t = [],[]
x,t = get_data(x,t)
for epoch in range(REPEAT_NUM):
    loss = 0
    total_loss = 0
    model.reset() # 勾配とメモリの初期化
    #print(epoch%10)
    for i in range(BATCH_SIZE):
        loss += model(x=np.array(x[index[epoch%EPOCH_NUM] + i],dtype="float32").reshape(1,in_size), t=np.array(t[index[epoch%EPOCH_NUM] + i],dtype="float32").reshape(1,out_size), train=True)
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
test_path = "C:/Users/odyssey8942/Documents/new_get_car_data/TeacherData3/control_myself_cardata.csv" 
predict_path = "G:/gr/cuda/test1/PredictData/Spin/"
p_file_name = str(in_size) + "_" + str(HIDDEN_SIZE) + "_" + str(out_size) + "_" + str(REPEAT_NUM) + "_predict.txt"
t_file_name = str(in_size) + "_" + str(HIDDEN_SIZE) + "_" + str(out_size) + "_" + str(REPEAT_NUM) + "_train.txt"

csv_file = open(test_path, "r", encoding="utf_8", errors="", newline="\n" )
f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
f_t = open(predict_path + t_file_name,"w")
f_p = open(predict_path + p_file_name,"w")

for row in f:
    obj = row[2:8]
    del obj[3]
    test_x.append(obj)
    test_t.append(row[13:14])
del test_x[len(test_x) - 10:]
del test_t[0:10]
test_x = np.array(test_x, dtype="float32")
test_t = np.array(test_t, dtype="float32")

test_x_zscore = zscore(test_x)
print(test_x)
print(test_x_zscore)

for x in test_x_zscore:
    x=x.reshape(1,in_size)
    y = model(x=x,train=False)
    predict = np.append(predict, y)

np.savetxt(f_p,predict)
np.savetxt(f_t,test_t)

serializers.save_npz(str(in_size) + "_" + str(HIDDEN_SIZE) + "_" + str(out_size) + "_" + str(REPEAT_NUM) + "_spin_model.npz",model)

N = len(test_t)
plt.plot(range(N), test_t, color="red", label="t")
plt.plot(range(N), predict, color="blue", label="y")
plt.legend(loc="upper left")
plt.show()