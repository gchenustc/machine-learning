import enum
import os
import numpy as np
import pickle
from importdata import *
from sklearn.model_selection import train_test_split
from collections import OrderedDict


def creatVocablist(docs):
    """
    创建语料表
    """
    vocab_set = set()
    for doc in docs:
        vocab_set = vocab_set | set(doc)
    return list(vocab_set)


def word2Vec(vocab_table, doc):
    """将单词对照着语料表转化为字典"""
    vocab_list = [0]*len(vocab_table)
    for word in doc:
        vocab_list[vocab_table.index(word)] += 1

    return vocab_list


def train(X_train, y_train):
    n_samples = len(X_train)
    n_words = len(X_train[0])
    p1 = sum(y_train=='spam') / float(n_samples) # 垃圾邮件概率
    
    p0Num = np.ones((n_words)) # 所有正常邮件的各个单词个数之和，先做一格平滑处理
    p1Num = np.ones((n_words))
    
    p0Denom = 2 # 所有正常邮件的所有单词之和，为分母，2是类别数，为拉普拉斯平滑处理
    p1Denom = 2
    
    for index,x in enumerate(X_train):
        if y_train[index] == 'ham':
            p0Num += x
            p0Denom += sum(x)
        else:
            p1Num += list(x)
            p1Denom += sum(x)
    
    # 词出现概率, log是放大作用
    p0_vec = np.log(p0Num / p0Denom)
    p1_vec = np.log(p1Num / p1Denom)

    return p0_vec,p1_vec,p1


def predict(X, p0_vec, p1_vec, p1):
    p1_class = np.log(p1) + sum(p1_vec * (X > 0))
    p0_class = np.log(1 - p1) + sum(p0_vec * (X > 0))
    return 'spam' if p1_class > p0_class else 'ham'


if __name__ == "__main__":
    # 导入数据
    sep = "/"
    importDataset() # 导入数据
    filename = "datasets"
    current_dir = os.path.dirname(__file__)
    with open(f"{current_dir}{sep}{filename}",'rb') as f:
        email = pickle.load(f)
    
    docs = email['data']
    y = email['labels']

    vocab_table = creatVocablist(docs)  # 创建语料表
    X = np.array([word2Vec(vocab_table, doc) for doc in docs])
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
    
    p0_vec,p1_vec,p1 = train(X_train, y_train)
    
    y_pre = np.array([predict(x,p0_vec, p1_vec, p1) for x in X_test])
    accu = sum(y_pre == y_test) / float(len(y_test))
    print(accu)
