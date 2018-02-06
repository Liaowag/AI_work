import numpy as np
import matplotlib.pyplot as plt


class Gene:
    def __init__(self, Distance, alpha, Path=None):
        '''
        初始化一个基因的参数，生成一个基因的初始路径，即其代表的解
        :param Distance:
        :param alpha: 一次交换次数
        :param Path:
        '''
        self.Distance = Distance
        self.num_cities = len(Distance)
        self.alpha = alpha
        # 随机生成初始的路径
        if Path is None:
            self.Path = list(range(self.num_cities))
            np.random.shuffle(self.Path)
        else:
            self.Path = Path.copy()

        self.Length = self._Length(self.Path)
        self.score = self._Score(self.Length)



    def _Length(self, Path):
        '''
        计算路径的总长度
        :param Path: 当前的路径
        :return: 长度
        '''
        Length = 0
        for i in range(self.num_cities - 1):
            Length += self.Distance[Path[i], Path[i + 1]]
        Length += self.Distance[Path[-1], Path[0]]
        return Length

    def _Score(self, Length):
        '''
        评估函数，根据路径的总长度进行评估
        :param Length:
        :return:
        '''
        return 1 / np.power(Length, self.alpha)


class GA:
    def __init__(self, cities, num_genes=90, num_children=90):
        '''
        初始化GA算法的参数
        :param cities: 传入城市节点
        :param num_genes: 父代个数
        :param num_children: 子代个数
        '''

        self.cities = cities
        self.num_cities = len(cities)
        self.Distance = np.empty((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                V = (cities[i][0] - cities[j][0],
                          cities[i][1] - cities[j][1])
                self.Distance[i, j] = np.linalg.norm(V)

        self.num_genes = num_genes
        self.num_children = num_children
        #每一次迭代的交换次数
        self.alpha = 50
        self.recombination_prob = 0.9

        #默认每次都进行突变
        self.mutation_prob = 1

        # 生成了父代的基因序列集合
        self.genes = [Gene(self.Distance, self.alpha) for i in range(self.num_genes)]

    def _turePath(self, Path1, Path2, s, t):
        '''
        对于基因重组之后的个体，需要进行调整，避免出现重复的进行
        :param Path1: 第一个体对应的路径
        :param Path2: 第二个个体对应的路径
        :param s:   开始交换节点
        :param t:   末尾交换节点
        :return:
        '''
        while len(set(Path1)) < self.num_cities:
            for i in list(range(0, s)) + list(range(t + 1, self.num_cities)):
                if Path1.count(Path1[i]) > 1:
                    p1 = i
                    break
            p2 = Path1.index(Path1[p1], s, t + 1)
            Path1[p1] = Path2[p2]

    def _Combine(self, Path1, Path2):
        '''
        基因组合函数
        :param Path1:  第一个体对应的路径
        :param Path2:  第二个个体对应的路径
        :return:
        '''
        Path1 = Path1.copy()
        Path2 = Path2.copy()

        if np.random.random() < self.recombination_prob:
            s, t = np.random.choice(range(1, self.num_cities - 1),
                                    size=2, replace=False)
            s, t = min(s, t), max(s, t)
            Path1[s:t + 1], Path2[s:t + 1] = Path2[s:t + 1], Path1[s:t + 1]
            self._turePath(Path1, Path2, s, t)
            self._turePath(Path2, Path1, s, t)
        return Path1, Path2

    def _Mutate(self, Path):
        '''
        突变函数
        :param Path:需要突变的基因个体
        :return:
        '''
        if np.random.random() < self.mutation_prob:
            p1, p2 = np.random.choice(range(self.num_cities), 2, replace=False)
            Path[p1], Path[p2] = Path[p2], Path[p1]

    def _Select(self, genes, number):
        '''
        轮盘赌算法选择出相应的父代基因
        :param genes: 传入基因
        :param number: 传入需要得到的子代个体
        :return:
        '''
        weights = [x.score for x in genes]
        sum_weight = sum(weights)
        weights = [x / sum_weight for x in weights]
        samples = np.random.choice(genes, number, p=weights, replace=False)
        return list(samples)

    def Iter_once(self):
        '''
        进化一次的过程
        :return:
        '''
        children = []
        for i in range(self.num_children // 2):
            parents = self._Select(self.genes, 10)
            parents.sort(key=lambda x: x.Length)
            #交叉组合
            Path1, Path2 = self._Combine(parents[0].Path, parents[1].Path)
            self._Mutate(Path1)
            self._Mutate(Path2)
            children.append(Gene(self.Distance, self.alpha, Path1))
            children.append(Gene(self.Distance, self.alpha, Path2))
        self.genes += children
        self.genes = self._Select(self.genes, self.num_genes)
        # 得到此次迭代的最好的基因
        bestgene = max(self.genes, key=lambda x: x.Length)
        return bestgene.Length, bestgene.Path

    def Iter(self, max_iter):
        '''
        整个算法的进化过程
        :param max_iter:  最大的迭代次数
        :return:
        '''
        self.genes = [Gene(self.Distance, self.alpha)
                      for i in range(self.num_genes)]
        bestgene = max(self.genes, key=lambda x: x.Length)
        bestLength, best_Path = bestgene.Length, bestgene.Path
        Length_list = [bestLength]
        bestItetate = 0

        #逐次迭代，并且当前最好的个体
        for i in range(1, max_iter + 1):
            Length, Path = self.Iter_once()
            Length_list.append(Length)
            print(f'Iterate: {i}, Lenigth: {Length:.4f}')
            if Length < bestLength:
                bestLength, best_Path = Length, Path
                bestItetate = i;

        print(f'Iterate times: {bestItetate}, Length: {bestLength}, Path: {best_Path}')
        self._Path(bestItetate, bestLength, best_Path)
        return bestLength, best_Path

    def _Path(self, iter_time, Length, Path):
        for i in range(-1, self.num_cities - 1):
            x = [self.cities[Path[i]][0], self.cities[Path[i + 1]][0]]
            y = [self.cities[Path[i]][1], self.cities[Path[i + 1]][1]]
            plt.plot(x, y)
        plt.title(f'Current Iterate times: {iter_time}, Length: {Length:.6f}')
        plt.show()




def Data(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = lines[0:]
    lines = lines[:-1]
    lines = [c.split(' ') for c in lines]
    cities = [(float(c[1]), float(c[2])) for c in lines]
    return cities


if __name__ == '__main__':
    cities = Data('GAtsp.tsp')
    ga = GA(cities, num_genes=110, num_children=110)
    ga.Iter(max_iter=500)
