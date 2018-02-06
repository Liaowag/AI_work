import numpy as np
import matplotlib.pyplot as plt


class ACO:
    def __init__(self, cities, num_ants=50):
        '''

        :param cities: 城市的数量，初始化为50
        :param num_ants: 蚂蚁的数量，初始化为50
        '''
        self.cities = cities
        self.num_cities = len(cities)

        #初始化距离矩阵
        self.Distance = np.empty((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                V = (cities[i][0] - cities[j][0], cities[i][1] - cities[j][1])
                self.Distance[i, j] = np.linalg.norm(V)
        self.Pher = np.ones((self.num_cities, self.num_cities))

        self.num_ants = num_ants
        self.alpha = 1
        self.beta = 2
        self.rho = 0.5
        self.Q = 100

        self.bestPath_Length = float('inf')
        self.bestPath = None





    def Iter(self, max_iter=200):
        '''
        算法的主要过程
        :param max_iter: 最大的迭代次数
        :return:
        '''
        bestLength = float('inf')
        bestPath = None
        Length_list = []
        bestIterate=0
        for i in range(1, max_iter + 1):
            Length, path = self.Iter_once()
            Length_list.append(Length)
            print(f'Iterate times: {i}, Length: {Length:.6f}, Path: {path}')

            if Length < bestLength:
                bestLength, bestPath = Length, path
                bestIterate=i


        bestLength = min(Length_list)
        print(f'The best Path is:  Iterate:{bestIterate}, Length: {bestLength:6f}')
        self._Path(bestIterate,bestLength,bestPath)
        return bestLength,bestPath

    def Iter_once(self):
        '''
        一次迭代之后，更新信息素浓度，找到当前最优路径与当前最优解
        :return:  当前最优长度与对应的路径
        '''
        path_list = []
        bestLength = float('inf')
        bestPath = None
        for i in range(self.num_ants):
            path = self._Ant()
            Length = self._Length(path)
            path_list.append((Length, path))
            if Length < bestLength:
                bestLength, bestPath = Length, path
        self._UpdatePher(path_list)
        return bestLength, bestPath

    def _Path(self, Itertime, Length, path):
        for i in range(-1, self.num_cities - 1):
            x = [self.cities[path[i]][0], self.cities[path[i + 1]][0]]
            y = [self.cities[path[i]][1], self.cities[path[i + 1]][1]]
            plt.plot(x, y)
        plt.title(f' Iteration times: {Itertime},  The path Length: {Length:.6f}')
        plt.show()


    def _Ant(self):
        '''
        蚂蚁不断根据当前的城市节点选择下一个节点，直到走遍所有的节点，返回最终的路径
        :return:
        '''
        available_cities = list(range(self.num_cities))
        path = []

        current_city = np.random.choice(available_cities)
        available_cities.remove(current_city)
        path.append(current_city)

        for i in range(self.num_cities - 1):
            next_city = self._NextCity(current_city, available_cities)
            available_cities.remove(next_city)
            path.append(next_city)
            current_city = next_city

        return path


    def _UpdatePher(self, path_list):
        '''
        :param path_list: 一次迭代的路径集合
        :return:
        '''
        Phers = np.zeros((self.num_cities, self.num_cities))
        for path_Length, path in path_list:
            amount_Pher = self.Q / path_Length
            for i in range(-1, self.num_cities - 1):
                Phers[path[i], path[i + 1]] += amount_Pher
                Phers[path[i + 1], path[i]] += amount_Pher
        self.Pher = self.Pher * self.rho + Phers

    def _Length(self, path):
        '''

        :param path:  输入的路径
        :return: 返回值为这个路径对应的路径长度
        '''
        Length = 0.0
        for i in range(-1, self.num_cities - 1):
            Length += self.Distance[path[i]][path[i + 1]]
        return Length

    def _NextCity(self, current_city, available_cities):
        '''

        :param current_city: 当前蚂蚁所在的城市
        :param available_cities: 当前蚂蚁可以到达的城市集合
        :return: 返回蚂蚁选择的下一个城市
        '''
        Probs = []
        for i in available_cities:
            # 根据选择城市的概率公式，得到分子的信息素浓度的alpha次方与到下一个城市启发式信息的beta次方
            Pher = pow(self.Pher[current_city][i], self.alpha)
            Distance = pow(self.Distance[current_city][i], self.beta)
            Probs.append(Pher / Distance)
        sum_weight = sum(Probs)
        Probs = list(map(lambda x: x / sum_weight, Probs))
        #这里根据轮盘赌方法随机选择下一个城市
        next_city = np.random.choice(available_cities, p=Probs)
        return next_city




def Data(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = lines[0:]
    lines = lines[:-1]
    lines = [c.split(' ') for c in lines]
    cities = [(float(c[1]), float(c[2])) for c in lines]
    return cities


if __name__ == '__main__':
    cities = Data('ACOtsp.tsp')
    aco = ACO(cities, num_ants=50,)
    aco.Iter(max_iter=50)
