from re import S
import numpy as np
import pandas as pd
from funcs import *
from sklearn.metrics import mean_squared_error,mean_squared_log_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline 

class logisticRegression(object):
    """
    会自动添加偏置项
    """
    def __init__(self, polynomial_degree=1, standarscaler=True):
        self.polynomial_degree = polynomial_degree
        self.standarscaler = standarscaler
        self.train_cost_hist = [] # 训练过程中损失历史
    
    def fit(self, X, y, alpha=0.1, iters=1000):
        if X.ndim == 1:
            X = X.reshape(-1,1)
        if y.ndim == 1:
            y = y.reshape(-1,1)

        # 加偏置项和多项式
        self.poly_features = PolynomialFeatures(degree=self.polynomial_degree, include_bias=True)
        
        # 标准化
        if self.standarscaler:
            self.standard_scaler = StandardScaler()
            self.processing = Pipeline([('standardscaler',self.standard_scaler),('polyfeatures',self.poly_features)])
            self.X = self.processing.fit_transform(X)
        else:
            self.X = self.poly_features.fit_transform(X)
            
        self.y = y
        
        self.num_train_samples = self.X.shape[0]
        self.num_train_features = self.X.shape[1]
        
        self.theta = np.random.randn(self.num_train_features,1)
        # print(self.theta)

        for i in range(iters):
            grads = self.numerical_gradient(self.X, self.y)
            self.theta -= alpha * grads
            self.train_cost_hist.append(self.cost(self.X, self.y))
        
    def hypothesis(self, X):
        """预测 - X已经经过标准化操作"""
        linear = X.dot(self.theta)
        return sigmoid(linear)
    
    def cost(self, X, t):
        """传入被处理后的特征的损失，X,t都是二维数组，X是特征，已经被处理过(多项式标准化等)，t是监督数据"""
        y_pre = self.hypothesis(X)
        return mean_squared_error(y_pre, t)
    
    
    def numerical_gradient(self, X, t):
        """计算梯度，训练时用, X是被处理过的特征"""
        cost_function = lambda theta: self.cost(X,t)
        grads = numerical_gradient_2d(cost_function, self.theta)
        return grads
    
    @staticmethod
    def accuracy(t,y_pre):
        """t:监督数据"""
        assert len(t) == len(y_pre)
        return np.sum(y_pre == t)/len(y_pre)

    def train_accuracy(self):
        """t:监督数据"""
        y = self.hypothesis(self.X)
        y_pre = np.where(y>0.5,1,0)
        return logisticRegression.accuracy(self.y, y_pre)

    def predict(self, X):
        """预测 - X未被标准化"""
        if X.ndim == 1:
            X = X.reshape(-1,1)
        if self.standarscaler:
            X = self.processing.transform(X)
        else:
            X = self.poly_features.transform(X)
        y = self.hypothesis(X)
        return np.where(y>0.5,1,0)

    def predict_probability(self, X):
        """预测 - X未被标准化 - 返回的是概率"""
        if X.ndim == 1:
            X = X.reshape(-1,1)
        if self.standarscaler:
            X = self.processing.transform(X)
        else:
            X = self.poly_features.transform(X)
        return self.hypothesis(X)

    def loss(self, X, t):
        """传入未被处理的特征的损失"""
        if X.ndim == 1:
            X = X.reshape(-1,1)
        if self.standarscaler:
            X = self.processing.transform(X)
        else:
            X = self.poly_features.transform(X)
            
        return self.cost(X,t)

def test_cost():
    y = np.array([[1],[0],[1]])
    t = np.array([[0.1],[0.1],[0.9]])
    print(mean_squared_error(y,t))


class LogisticRegressionMultiClass(object):
    """
    会自动添加偏置项
    """
    def __init__(self, polynomial_degree=1, standarscaler=True):
        self.polynomial_degree = polynomial_degree
        self.standarscaler = standarscaler
    
    def fit(self, X, y, alpha=0.1, iters=1000):
        self.X = np.copy(X)
        self.y = np.copy(y)
        self.num_samples = self.X.shape[0]
        
        self.unique_labels = np.unique(y)
        self.num_unique_labels = self.unique_labels.shape[0]
        
        self.theta = []
        # 储存每一个二分类逻辑回归类
        self.logistics = {}
        # 储存每一个二分类的标签
        self.labels = {}
        # 储存每个特征训练的cost
        self.train_cost_hist = {}
        
        for index,label in enumerate(self.unique_labels):
            self.labels[label] = np.where(self.y==label,1,0)
            self.logistics[label] = logisticRegression(self.polynomial_degree, self.standarscaler)
            self.logistics[label].fit(self.X, self.labels[label],alpha=alpha, iters=iters)
            self.train_cost_hist[label] = self.logistics[label].train_cost_hist
            self.theta.append(self.logistics[label].theta.flatten())
        self.theta = np.array(self.theta)
        # print(self.theta)
        
    def predict(self,X):
        """特征X未被预处理过"""
        num_samples = X.shape[0]
        result_probability = np.zeros((num_samples, self.num_unique_labels))
        for index,label in enumerate(self.unique_labels):
            result_probability[:,index:index+1] = self.logistics[label].predict_probability(X)
        # print(result_probability)
        max_prob_index = np.argmax(result_probability,axis=1)
        class_pred = np.empty(max_prob_index.shape, dtype=object)
        # print(max_prob_index)
        for index,label in enumerate(self.unique_labels):
            class_pred[max_prob_index==index] = label
        return class_pred.reshape(num_samples, 1)
    
    @staticmethod
    def accuracy(t,y_pre):
        """t:监督数据"""
        assert len(t) == len(y_pre)
        return np.sum(y_pre == t)/len(y_pre)
    
    
            


def test():
    data = pd.read_csv('data/iris.csv')

    # 选取两个特征 - 方便绘图
    x_axis = 'petal_length'
    y_axis = 'petal_width'
    #　切分测试集和训练集
    train_data = data.sample(frac = 0.8)
    test_data = data.drop(train_data.index)
    X_train = train_data[[x_axis,y_axis]].values
    y_train = train_data[['class']].values
    X_test = test_data[[x_axis,y_axis]].values
    y_test = test_data[['class']].values
    ls = LogisticRegressionMultiClass()
    ls.fit(X_train, y_train)
    y_test_pre = ls.predict(X_test)
    print(ls.accuracy(y_test,y_test_pre))
    y_train_pre = ls.predict(X_train)
    print(ls.accuracy(y_train,y_train_pre))

# test()