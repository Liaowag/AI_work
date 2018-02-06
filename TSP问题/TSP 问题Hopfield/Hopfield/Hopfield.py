import numpy as np
import matplotlib.pyplot as plt


class Hopfield:
    def __init__(self, cities):
        '''
        网络初始化
        :param cities: 城市坐标集合
        '''
        self.cities = cities
        self.num_cities = len(cities)

        self.U0 = 0.02  # 初始变化率
        self.alpha = 1e-6
        self.size_turePath = 1.5

        #随机初始化矩阵网络参数
        Network = (self.num_cities, self.num_cities)
        self.Distance = np.empty(Network)
        for i in range(self.num_cities):
            for j in range(i, self.num_cities):
                V = (cities[i][0] - cities[j][0],
                          cities[i][1] - cities[j][1])
                #更新距离集合
                self.Distance[i, j] = np.linalg.norm(V)
                self.Distance[j, i] = self.Distance[i, j]
        self.actual_Distance = self.Distance
        self.Distance = self.Distance / self.Distance.max()

        #初始化网络输入
        self.U = np.ones(Network)
        self.U /= self.num_cities ** 2
        #得到随机的δ值
        self.U += np.random.uniform(-0.5, 0.5, Network) / 10000
        self.delta_U = np.zeros(Network)
        #初始化输出
        self.V = self._Sigmoid(self.U)




    def Iter(self, max_iter):
        '''
        迭代函数
        :param max_iter: 最大迭代次数
        :return:
        '''

        #每迭代300次输出迭代的矩阵图像
        iter = 300
        for i in range(1, max_iter + 1):
            self.Iter_once()
            if iter and i % iter == 0:

                self._Mat(self.V)

        Path = np.array(self.V).argmax(0)
        if iter:
            self._Path(Path)
        return Path.tolist()


    def Iter_once(self):
        '''
        迭代一次函数，更新输入和输出值
        :return:
        '''
        self.delta_U = np.zeros((self.num_cities, self.num_cities))
        for city in range(self.num_cities):
            for p in range(self.num_cities):
                self.delta_U[city, p] = \
                    self.alpha * self._Delta(city, p)
        self.U += self.delta_U
        self.V = self._Sigmoid(self.U)


    def _Mat(self, mat):
        '''
        画出矩阵函数
        :param mat: 得到的迭代的矩阵
        :return:
        '''
        plt.imshow(mat, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        plt.show()

    def _Path(self, Path):
        '''
        画出路径函数
        :param Path: 路径的集合
        :return:
        '''
        length = 0
        for i in range(-1, self.num_cities - 1):
            length += self.actual_Distance[Path[i], Path[i + 1]]
            x = [self.cities[Path[i]][0], self.cities[Path[i + 1]][0]]
            y = [self.cities[Path[i]][1], self.cities[Path[i + 1]][1]]
            plt.plot(x, y)
        print (f'The Length is: {length:.6f} ')
        plt.title(f'{length:.6f}')
        plt.show()
    def _Delta(self, city, p):
        '''
        求∆值的函数，这里是近似使用参数逼近的
        :param city:
        :param p:
        :return:
        '''

        delta = -self.U[city, p]
        value = np.sum(self.V[city, :])
        value -= self.V[city, p]
        delta -= value * 500
        value = np.sum(self.V[:, p])
        value -= self.V[city, p]
        delta -= value * 500
        value = np.sum(self.V)
        value -= self.num_cities + self.size_turePath
        delta -= value * 200
        value = 0.0
        for x in range(self.num_cities):
            left = self.V[x, p - 1]
            right = self.V[x, (p + 1) % self.num_cities]
            value += self.Distance[city, x] * (right + left)
        delta -= value * 500
        return delta


    def _Sigmoid(self, u):
        '''
        激活函数
        :param u: 设定的参数
        :return: 函数值
        '''
        return 0.5 * (1 + np.tanh(u / self.U0))




def Data(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = lines[0:]
    lines = lines[:-1]
    lines = [c.split(' ') for c in lines]
    cities = [(float(c[1]), float(c[2])) for c in lines]
    return cities


if __name__ == '__main__':
    cities = Data('Hopfieldtsp.tsp')
    H = Hopfield(cities)
    Path = H.Iter(2400)
    print(f'The Path is: {Path}')

