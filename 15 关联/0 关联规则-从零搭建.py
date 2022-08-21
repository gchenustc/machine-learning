from locale import currency
import numpy as np
import pandas as pd
import os

def loadDataSet():
    return [[1,3,4],
            [2,3,5],
            [1,2,3,5],
            [2,5]]
    

def createC1(dataset):
    C1 = []
    for transaction in dataset:
        for item in transaction:
            if [item] not in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset,C1))


def scanData(dataset, Ck, min_support=0.5):
    """
    Ck: k项集 -- 如C1 [{1},{2},{3},...]
    """
    support_data = {}
    sup_denominator = len(dataset)
    for transaction in dataset:
        for item in Ck:
            if item.issubset(set(transaction)):
                if not item in support_data:  # item 不能是哈希的（no set, frozeset can.） - 字典的键不能是哈希的
                    support_data[item] = 1
                else:
                    support_data[item] += 1
    
    retlist = []
    for item in support_data:
        support_data[item] /= sup_denominator
        if support_data[item] >= min_support:
            retlist.append(item)
    
    return retlist, support_data


def aprioriGen(C, k):
    """
    拼接 - 一项集拼成二项集，以此类推
    C 是 k-1 项集，k是目标为k的项集数
    """
    retlist = []
    lenc = len(C)
    for i in range(lenc):
        for j in range(lenc):
            item = C[i].union(C[j]) # 返回的是拷贝
            if len(item)==k and item not in retlist: # 第一个判断是去除重复
                retlist.append(item)
    return retlist


def apriori(dataset, min_support=0.5):
    support_data = dict()
    C_total = []
    Ck = createC1(dataset) # C1

    k = 2
    while True:
        Ck_scan, support_data_son = scanData(dataset, Ck, min_support=min_support)
        if not Ck_scan: break
        C_total.append(Ck_scan)
        Ck = aprioriGen(Ck_scan, k)
        support_data.update(support_data_son)
        k += 1
    
    return C_total, support_data


def rules(fren_item, support_data, min_conf=0.5, min_lift=0.5):
    transaction = []
    conf = []
    lift = []
    antecedent_support = []
    consequent_support = []
    for i in range(1,len(fren_item)): # i项集
        for j in range(0,i): # i --> j
            for item_i in fren_item[i]:
                for item_j in fren_item[j]:
                    if item_j.issubset(item_i):
                        conf_ = support_data[item_i] / support_data[item_i-item_j]
                        lift_ = conf_ / support_data[item_j]
                        transaction.append(f"{set(item_i-item_j)}-->{set(item_j)}")
                        conf.append(conf_)
                        lift.append(lift_)
                        antecedent_support.append(support_data[item_i])
                        consequent_support.append(support_data[item_j])
    ret = pd.DataFrame({"transaction":transaction,
                    "antecedent_support":antecedent_support,
                    "consequent_support":consequent_support,
                    "confidence":conf,
                    "lift":lift})
    return ret[(ret["confidence"]>min_conf) & (ret["lift"]>min_lift)]

currdir = os.path.dirname(__file__)
data = pd.read_csv(f"{currdir}/datasets/ml-10M100K/movies.dat",sep="::",names=["index","title","genres"],engine="python")
data_preprocess = data.genres.tolist()
dataset = list(map(lambda x: x.split("|"),data_preprocess))
fren_item,support_data = apriori(dataset,min_support=0.01)
print(rules(fren_item, support_data, min_conf=0.3, min_lift=4))