import numpy as np
from collections import OrderedDict

# 下面函数用来调用
import numpy as np

# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """输入一维数组或者二维数组，输入一维返回一维，输入二维返回二维"""
    if x.ndim == 2: # x=[[1,2,3],[2,3,4]]
        x = x.T # x=[[1,2],[2,3],[3,4]]
        x = x - np.max(x,axis=0) # x=x - [3,4] = [[1-3,2-4],[2-3,3-4],[3-3,4-4]] = [[-2,-2],[-1,-1],[0,0]]
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T # 返回二维数组
    # x.ndim == 1 的情况
    x = x-np.max(x)
    return np.exp(x) / np.sum(np.exp(x)) # 返回一维数组比如,ret.shape=(3,)

# 梯度
# 下面两个函数定义了多个数据的梯度，其中每个数据可以是一元或者多元函数
def _numerical_gradient_1d(f,x):
    """
    求一元或者多元函数梯度，传入的 x 是一维数组，代表坐标，浮点数。比如 x = [1.0,2.0] 就是二元函数在 (1,2) 上的点。求的是在这个点上的二元函数的两个方向的偏导
    """
    h = 1e-4

    grad = np.zeros_like(x) # 假如是二元函数，传入变量 x = [3,4]，则现在 grad = [0,0]，grad[0],grad[1] 分别是二元函数的两个变量的梯度

    for idx in range(x.size): # x: [3,4], idx: [0,1]
        tmp_val = x[idx] # tmp_val=x[0]=3
        x[idx] = tmp_val + h # x: [3+h,4]
        fxh1 = f(x) # [3+h,4] 对应的函数值 f
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1-fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原x
        
    return grad
    

def numerical_gradient_2d(f,X):
    """2d数组的梯度"""

    if X.ndim == 1:
        return _numerical_gradient_1d(f,X)
    else:
        grad = np.zeros_like(X) # X=[[2,3,4],[1,2,1]], grad=[[0,0,0],[0,0,0]]
        
        for idx, x in enumerate(X): #  x=[2,3,4],[1,2,1], idx=0,1
            grad[idx] = _numerical_gradient_1d(f,x)
        
        return grad

# 梯度下降函数
def gradient_descent(f, init_x, lr=0.01, step_num=300):
    x = init_x # 假设是二元函数，x=[2,2], f=x[0]**2+x[1]**2
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())
        
        grad = numerical_gradient_2d(f,x)  # grad=[4,4]
        x -= lr * grad  # x = [2,2] - 0.01*[4,4] = [1.96,1.96]

    return x, np.array(x_history)

# softmax测试
def test_softmax():

    arr = np.array([0.2,0.3])
    print(softmax(arr)) # [0.47502081 0.52497919]

    arr = np.array([2,4])
    print(softmax(arr)) # [0.11920292 0.88079708]
