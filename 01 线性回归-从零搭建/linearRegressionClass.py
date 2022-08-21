import numpy as np
from cmath import sin
from prepare import *

# 早期版本的测试
class LinearRegressionTest(object):
    def __init__(self, data, label, polynomial_degree=0, sinusoid_degree=0, normalize=True):
        """
        data是一维(单个特征)或者二维数组(单个特征，多个特征)，label是一维数组([0,1,1])者二维数组([[0],[1],[1]])
        """
        # 如果 data 只有一格特征并且是一维，转换为二维
        if data.ndim == 1:
            data = data.reshape((-1,1))
        
        # 如果 label 是二维数组，转换为一维
        if label.ndim == 2:
            label = label.reshape((-1,))

        # 初始化特征
        # 归一化（可选）,最后加一列，polynomial or sinusoid 转换
        self.data = prepare_for_training(self.data, self.polynomial_degree, self.sinusoid_degree, self.normalize)[0]
        self.label = np.copy(label)

        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize = normalize
        # print(self.data)
        # print(self.label)
        # print(self.data.shape,self.label.shape)


        # 训练数据的数据数 与 特征数
        self.num_examples, self.num_features = self.data.shape

        # 初始化 theta
        self.theta = np.zeros((self.num_features,)) # 如果有5维特征，self.theta = [0,0,0,0,0]
        # self.theta = np.zeros((self.data.shape[1],1)) 
        # print(self.theta)

    def train(self, alpha = 0.01, num_iters = 500):
        """
        alpha: 学习率
        num_iters: 迭代次数
        """
        loss_hist = []
        for _ in range(num_iters):
            self.theta -= alpha * self.gradient()
            loss_ = self.loss(self.data, self.label)
            print(loss_,self.theta)
            loss_hist.append((loss_))

        return loss_hist

    
    def loss(self, data, label):
        """计算loss之前data要被处理过 - normalize等"""
        if data.ndim == 1:
            data = data[None,:]
        if label.ndim == 2:
            label = label.flatten()
        num_examples = data.shape[0]
        # print(data.shape,label.shape,self.theta.shape)
        return 1 / (2 * num_examples) * np.sum((label - np.dot(data, self.theta))**2)

    def gradient(self):
        grads = - 1 / self.num_examples * np.dot(self.data.T, self.label - self.hypothesis(self.data))
        return grads

    
    def hypothesis(self, data):
        """
        data: 一维或者二维
        theta: 一维或者二维
        如果 data 和 theta 都是二维，返回的是二维(n,1)数组，否则返回一维数组
        """
        return np.dot(data,self.theta)
    
    def predict(self, data):
        data_processed = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize)[0]
        return np.dot(data_processed ,self.theta)


class LinearRegression(object):
    
    def __init__(self, polynomial_degree=0, sinusoid_degree=0, nomalize_data = True):
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.nomalize_data = nomalize_data
        
    def fit(self, data, labels, alpha=0.01, num_iters=500):
        """
        传入参数，和对参数的处理
        data 如果只有一个特征可以传入一维数组，label也\
        可以传入一维数组，但 data 和 labels (labels.shape=(n,1)) 最终需要转换为二维数组的形式
        """
        # 导入数据
        # 转换训练数据为正确格式
        if data.ndim == 1:  # 传入的数据特征只有一格并且是一维
            self.data = data.reshape((-1,1))  # reshape 返回的是拷贝
        else:
            self.data = np.copy(data)

        # 数据预处理
        self.data = prepare_for_training(self.data, self.polynomial_degree, self.sinusoid_degree, self.nomalize_data)[0]
            
        if labels.ndim == 1:
            self.labels = labels.reshape((-1,1))
        else:
            self.labels = np.copy(labels)

        self.num_examples, self.num_features =  self.data.shape
        
        # theta 的个数和 features 个数相同
        # theta 是二维数组
        self.theta = np.zeros((self.num_features,1))
        # print(self.data.shape, self.theta.shape)
        self.train(alpha, num_iters)
        
    def train(self, alpha, num_iters):
        self.cost_hist = []
        for _ in range(num_iters):
            # 每次循环都执行一次梯度下降，同时更新self.theta
            self.gradient_step(alpha) # 执行完梯度下降后，self.theta会被更新
            self.cost_hist.append(self.cost(self.data, self.labels))
    
    def cost_history(self):
        """
        返回训练过程中的cost记录 - 以列表的形式返回
        """
        return self.cost_hist
    
    def gradient_step(self, alpha):
        """
        梯度下降一次并且更新self.theta
        """
        predictions = self.hypothesis(self.data)
        delta = self.labels - predictions
        self.theta += alpha * (1 / self.num_examples) * np.dot(delta.T, self.data).T
    
    def hypothesis(self, data):
        """
        传入特征(data)， 通过self.theta返回预测值
        特征数据(data)已经经过预处理
        """ 
        return np.dot(data, self.theta) # 返回的是预测值 - 二维数组(shape = (n,1))
    
    def cost(self,data,labels):
        """
        计算cost, 特征数据(data)和标签(labels)已经经过预处理
        """
        num_examples = data.shape[0]
        
        delta = labels - self.hypothesis(self.data)
        cost = 1/(2*num_examples) * np.dot(delta.T, delta)  # delta.shape=(n,1), delta.T.shape=(1,n), np.dot(delta.T, delta).shape=(1,1)
        return cost[0][0]
    
    def predict(self, data):
        """
        已经训练过了，预测新的数据
        """
        if data.ndim == 1:  # 转换为二维数组
            data = data.reshape((-1,1))
        data_processed = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.nomalize_data)[0]
        return self.hypothesis(data_processed)  # 返回的是二维数组 shape = (num_features, 1)

    def get_cost(self, data, labels):
        if data.ndim == 1:  # 转换为二维数组
            data = data.reshape((-1,1))
        if labels.ndim == 1:
            labels = labels.reshape((-1,1))
            
        data_processed = prepare_for_training(self.data, self.polynomial_degree, self.sinusoid_degree, self.nomalize_data)[0]
        return self.cost(data_processed, labels)
        


# 唐宇迪
class LinearRegression_:

    def __init__(self,data,labels,polynomial_degree = 0,sinusoid_degree = 0,normalize_data=True):
        """
        1.对数据进行预处理操作
        2.先得到所有的特征个数
        3.初始化参数矩阵
        """
        (data_processed,
         features_mean, 
         features_deviation)  = prepare_for_training(data, polynomial_degree, sinusoid_degree,normalize_data=True)
         
        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data
        
        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features,1))
        
    def train(self,alpha=0.01,num_iterations = 500):
        """
                    训练模块，执行梯度下降
        """
        cost_history = self.gradient_descent(alpha,num_iterations)
        return self.theta,cost_history
        
    def gradient_descent(self,alpha,num_iterations):
        """
                    实际迭代模块，会迭代num_iterations次
        """
        cost_history = []
        for _ in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data,self.labels))
        return cost_history
        
        
    def gradient_step(self,alpha):    
        """
                    梯度下降参数更新计算方法，注意是矩阵运算
        """
        num_examples = self.data.shape[0]
        prediction = LinearRegression_.hypothesis(self.data,self.theta)
        delta = prediction - self.labels
        theta = self.theta
        theta = theta - alpha*(1/num_examples)*(np.dot(delta.T,self.data)).T
        self.theta = theta
        
        
    def cost_function(self,data,labels):
        """
                    损失计算方法
        """
        num_examples = data.shape[0]
        delta = LinearRegression_.hypothesis(self.data,self.theta) - labels
        cost = (1/2)*np.dot(delta.T,delta)/num_examples
        return cost[0][0]
        
        
        
    @staticmethod
    def hypothesis(data,theta):   
        predictions = np.dot(data,theta)
        return predictions
        
    def get_cost(self,data,labels):  
        data_processed = prepare_for_training(data,
         self.polynomial_degree,
         self.sinusoid_degree,
         self.normalize_data
         )[0]
        
        return self.cost_function(data_processed,labels)

    def predict(self,data):
        """
                    用训练的参数模型，与预测得到回归值结果
        """
        data_processed = prepare_for_training(data,
         self.polynomial_degree,
         self.sinusoid_degree,
         self.normalize_data
         )[0]
         
        predictions = LinearRegression_.hypothesis(data_processed,self.theta)
        
        return predictions
