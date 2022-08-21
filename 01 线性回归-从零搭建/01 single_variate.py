import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from linearRegression import *

dirname = os.path.dirname(__file__)
data = pd.read_csv(dirname + "/data/world-happiness-report-2017.csv", sep=',')
train_data = data.sample(frac = 0.8)
test_data = data.drop(train_data.index)

# 选定考虑的特征
input_param_name = 'Economy..GDP.per.Capita.'
output_params_name = 'Happiness.Score'

x_train = train_data[[input_param_name]]
y_train = train_data[[output_params_name]]
x_test = test_data[[input_param_name]]
y_test = test_data[[output_params_name]]


# 用创建的随机样本测试
    # 构造样本的函数
# def fun(x, slope, noise=1):
#     x = x.flatten()
#     y = slope*x + noise * np.random.randn(len(x))
#     return y

#     # 构造数据
# slope=2
# x_max = 10
# noise = 0.1
# x_train = np.arange(0,x_max,0.2).reshape((-1,1))
# y_train = fun(x_train, slope=slope, noise=noise)
# x_test = np.arange(x_max/2, x_max*3/2, 0.2).reshape((-1,1))
# y_test = fun(x_test, slope=slope, noise=noise)

#     #观察训练样本和测试样本
# # plt.scatter(x_train, y_train, label='train data', c='b')
# # plt.scatter(x_test, y_test, label='test data', c='k')
# # plt.legend()
# # plt.title('happiness - GDP')
# # plt.show()


#     #测试 - 与唐宇迪的对比
# lr = LinearRegression()
# lr.fit(x_train, y_train)
# print(lr.predict(x_test))
# print(y_test)

# y_train = y_train.reshape((-1,1))
# lr = LinearRegression_(x_train, y_train)
# lr.train()
# print(lr.predict(x_test))
# print(y_test)


lr = LinearRegression()
lr.fit(x_train, y_train, alpha=0.01, num_iters=500)
y_pre = lr.predict(x_test)
print("开始损失和结束损失",lr.cost_hist[0],lr.cost_hist[-1])
# iters-cost curve
# plt.plot(range(len(lr.cost_hist)), lr.cost_hist)
# plt.xlabel('Iter')
# plt.ylabel('cost')
# plt.title('GD')
# plt.show()
plt.scatter(x_train,y_train,label='Train data')
plt.scatter(x_test,y_test,label='test data')
plt.plot(x_test, y_pre,'r',label = 'Prediction')
plt.xlabel(input_param_name)
plt.ylabel(output_params_name)
plt.title('Happy')
plt.legend()
plt.show()