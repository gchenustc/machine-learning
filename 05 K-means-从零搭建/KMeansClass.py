import numpy as np

class KMeans(object):
    def __init__(self, n_clustres, max_iters=100):
        self.n_clustres = n_clustres
        self.max_iters = max_iters

    def fit(self, data):
        self.data = data
        self.n_examples, self.n_features = data.shape

        # 1. 初始化聚类中心 - 得到num_clustres个聚类中心
        self.cluster_centers = self.centroids_init()

        # 2. 得到样本点到最近聚类中心的距离
        self.closest_centroids_ids = np.empty((self.n_examples, 1))

        for _ in range(self.max_iters):
            # 3. 对每个样本点，找到最近的聚类中心
            self.closest_centroids_ids = self.find_closest_centroids()
            # 4. 更新中心点
            self.cluster_centers = self.update_centroids()
        
        return self.cluster_centers, self.closest_centroids_ids

    def update_centroids(self):
        centroids = np.zeros((self.n_clustres, self.n_features))
        
        for centroids_id in range(self.n_clustres):
            ids = (self.closest_centroids_ids == centroids_id).flatten()
            centroids[centroids_id] = np.mean(self.data[ids],axis=0)
        
        return centroids
        
    def find_closest_centroids(self):
        n_centroids = self.cluster_centers.shape[0]
        closest_centroids_ids = np.empty((self.n_examples, 1))

        for i in range(self.n_examples):
            distance = np.zeros((n_centroids, 1))
            for j in range(n_centroids):
                # 计算每个样本点到聚类中心的距离
                distance_diff = np.sum((self.data[i] - self.cluster_centers[j]) ** 2)
                distance[j] = distance_diff
            # 得到最近的聚类中心
            closest_centroids_ids[i] = np.argmin(distance)

        return closest_centroids_ids
                
    def centroids_init(self):
        # 随机选取聚类中心
        # 返回二维数组
        random_ids = np.random.permutation(self.n_examples)
        return self.data[random_ids[:self.n_clustres]]
