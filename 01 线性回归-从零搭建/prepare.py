import numpy as np


def normalize(features):
    """
    传入的数据格式是二维数组(一维其实也行) - 比如三特征四数据
    [[2,3,4],[1,2,3],[3,1,2],[2,3,1]]
    """
    # 防止分母为0而设置的参数
    h = 1e-10

    features_normalized = np.copy(features).astype(float)
    
    # 均值
    mean = np.mean(features, 0)  # # [1.75 2.25 2.5]
    std = np.std(features, 0)

    # print(mean,std)
    # print(features_normalized - mean)

    features_normalized = (features_normalized - mean) / (std+h)

    return features_normalized, mean, std


# 唐宇迪
def normalize_(features):

    features_normalized = np.copy(features).astype(float)

    # 计算均值
    features_mean = np.mean(features, 0)

    # 计算标准差
    features_deviation = np.std(features, 0)

    # 标准化操作
    if features.shape[0] > 1:
        features_normalized -= features_mean

    # 防止除以0
    features_deviation[features_deviation == 0] = 1
    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation


def generate_polynomials_one_feature(dataset, polynomial_degree, normalize_data=False):
    """变换方法：
    x1, x2, x1^2, x2^2, x1*x2, x1*x2^2, etc.
    """

    num_examples = dataset.shape[0]
    polynomials = np.empty((num_examples,0))

    for i in range(2, polynomial_degree + 1):
        polynomial_feature = (dataset ** i)
        polynomials = np.concatenate((polynomials, polynomial_feature), axis=1)

    if normalize_data:
        polynomials = normalize(polynomials)[0]

    return polynomials


def generate_polynomials(dataset, polynomial_degree, normalize_data=False):
    """
    变换方法: x1, x2, x1^2, x2^2, x1*x2, x1*x2^2, etc.
    polynomial_degree = 1 时 返回空
    dataset: 单特征（一维或者多维数组）或者多特征（二维数组）
    """

    # 如果是一维数组，转换为二维
    if dataset.ndim == 1:
        dataset = dataset.reshape(-1,1)

    # 获取样本数
    num_examples = dataset.shape[0]
        
    if dataset.shape[1] == 1:
        return generate_polynomials_one_feature(dataset, polynomial_degree, normalize_data)
    
    features_split = np.array_split(dataset, 2, axis=1)
    dataset_1 = features_split[0]
    dataset_2 = features_split[1]

    polynomials = np.empty((num_examples,0))

    for i in range(2, polynomial_degree + 1):
        for j in range(i + 1):
            polynomial_feature = (dataset_1 ** (i - j)) * (dataset_2 ** j)
            polynomials = np.concatenate((polynomials, polynomial_feature), axis=1)

    if normalize_data:
        polynomials = normalize(polynomials)[0]

    return polynomials

# features = np.array([[1,3,2],[2,4,1],[3,6,2]]) 
# generate_polynomials(features, 2, False)


def generate_sinusoids(data, degree, normalize_data = False):
    """sinx"""
    num_examples = data.shape[0]
    sinusoids = np.empty((num_examples,0))

    for degree in range(1, degree+1):
        sinusoids_features = np.sin(degree * data)
        sinusoids = np.concatenate((sinusoids,sinusoids_features), axis=1)

    if normalize_data:
        sinusoids = normalize(sinusoids)[0]

    return sinusoids


def prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
    """
    可以传入一维或者二维的训练特征数据，返回的都是二维特征
    polynomial_degree >= 2才有效果
    """

    data_processed = np.copy(data)

    if data_processed.ndim == 1:
        data_processed = data_processed.reshape((-1,1))

    mean,std = -1,-1
    if normalize_data:
        data_normalize,mean,std = normalize(data_processed)
        data_processed = data_normalize
    
    if sinusoid_degree > 0:
        sinusoid = generate_sinusoids(data_normalize, sinusoid_degree)
        data_processed = np.concatenate((data_processed, sinusoid),axis=1)

    if polynomial_degree > 1:
        polynomial = generate_polynomials(data_normalize, polynomial_degree)
        data_processed = np.concatenate((data_processed, polynomial),axis=1)

    # 在最后加一列
    data_processed = np.hstack((data_processed, np.ones((data_processed.shape[0], 1))))

    return data_processed,mean,std


# 唐宇迪
def prepare_for_training_(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):

    # 计算样本总数
    num_examples = data.shape[0]

    data_processed = np.copy(data)

    # 预处理
    features_mean = 0
    features_deviation = 0
    data_normalized = data_processed
    if normalize_data:
        (
            data_normalized,
            features_mean,
            features_deviation
        ) = normalize(data_processed)

        data_processed = data_normalized

    # 特征变换sinusoidal
    if sinusoid_degree > 0:
        sinusoids = generate_sinusoids(data_normalized, sinusoid_degree)
        data_processed = np.concatenate((data_processed, sinusoids), axis=1)

    # 特征变换polynomial
    if polynomial_degree > 0:
        polynomials = generate_polynomials(data_normalized, polynomial_degree, normalize_data)
        data_processed = np.concatenate((data_processed, polynomials), axis=1)

    # 加一列1
    data_processed = np.hstack((np.ones((num_examples, 1)), data_processed))

    return data_processed, features_mean, features_deviation


# features = np.array([[1,3,2],[2,4,1],[3,6,2]]) 
# a = prepare_for_training(features)
# b = prepare_for_training_(features)
# print(a,b)