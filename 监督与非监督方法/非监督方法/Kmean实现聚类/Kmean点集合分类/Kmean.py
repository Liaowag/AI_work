# coding:utf-8
import numpy
import random


class Kmean:
    """
   此处设定了两种模式，一种是不输入参数，那么随机生成点集，随机分类，当然随机产生的点的范围需要确定，分类的个数需要确定
   两外一种是输入数据集和分类的个数，进行实验
    """

    @classmethod
    def __init__(self, dataset=None, centroid=None):
        '''

        :param dataset: 传入的点的数据集合
        :param centroid:  指定的初始的均值点的个数
        '''
        self.dataset = dataset
        self.centroid = centroid


    @classmethod
    def get_centroid_numbers(self):
        '''
        分类的类数
        :return:
        '''
        return 8


    @classmethod
    def get_random_points(self, numbers, begin=1, end=10):
        '''
        生成了随机的点集合
        :param numbers: 生成的点集合的数目
        :param begin: 范围下限
        :param end: 范围上限
        :return:
        '''
        dataset = [[random.uniform(begin, end), random.uniform(begin, end)] for i in range(numbers)]
        dataset = [[float("{0:.2f}".format(data)) for data in point] for point in dataset]
        return dataset



    @classmethod
    def get_Distance(self, point_a, point_b):
        '''
        得到两个点之间的距离
        :param point_a:
        :param point_b:
        :return:
        '''
        return numpy.sqrt(sum((numpy.array(point_a) - numpy.array(point_b)) ** 2))


    @classmethod
    def get_clusters_points(self, dataset, centroids):
        '''
        将所有的点进行分类，分为K类
        :param dataset: 数据集合
        :param centroids: 均值点集合
        :return:
        '''
        clusters_points = []
        for i in range(len(centroids)):
            clusters_points.append([])
        for point in dataset:
            Distances = [self.get_Distance(point, centroid) for centroid in centroids]
            belonging_cluster_id = Distances.index(min(Distances))
        # 进行点的分类划分
            clusters_points[belonging_cluster_id].append(point)
        return clusters_points



    @classmethod
    def move_centroids(self, clusters_points):
        '''
        得到了均值点的集合
        :param clusters_points: 均值点集合
        :return:
        '''
        mu = []
        for points in clusters_points:
            mu.append(numpy.mean(points, axis=0))
        return mu



    @classmethod
    def converged(self, old_mu, new_mu):
        '''
        检查前后两次分类迭代时候发生变化，这是迭代的停止条件
        :param old_mu: 原分类
        :param new_mu: 新分类
        :return:
        '''
        for (i, k) in zip(old_mu, new_mu):
            for (x, y) in zip(i, k):
                if x != y: return False
        return True



    @classmethod
    def get_clusters(self):
        if self.dataset == None:
            self.dataset = self.get_random_points(100)
        if self.centroid == None:
            self.centroid = self.get_centroid_numbers()
        # 在范围中随机生成初始的中心点
        mu = self.get_random_points(self.centroid)
        last_mu = self.get_random_points(self.centroid)
        # 避免首先选择两个集合的情况
        first_loop = True
        while first_loop or not self.converged(last_mu, mu):
            if first_loop: first_loop = False
            last_mu = mu
            # 分类
            clusters_points = self.get_clusters_points(self.dataset, mu)
            mu = self.move_centroids(clusters_points)
        return (mu, clusters_points)

