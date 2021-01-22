#  #################################################################
#  This file contains the main DROO operations, including building DNN, 
#  Storing data sample, Training DNN, and generating quantized binary offloading decisions.

#  version 1.0 -- January 2020. Written based on Tensorflow 2 by Weijian Pan and 
#  Liang Huang (lianghuang AT zjut.edu.cn)
#  #################################################################

from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print(tf.__version__)
print(tf.keras.__version__)


# DNN network for memory
class MemoryDNN:
    def __init__(
            self,
            net,
            learning_rate=0.01,
            training_interval=10,  # 表示没存储多少个经验就可以进行训练了
            batch_size=100,
            memory_size=1000,  # 最多存储的经验数据的大小
            output_graph=False
    ):
        '''
        Args:
            net (array): 分别表示DNN每层的神经元个数
            learning_rate (float): 学习率
            training_interval (int): 表示每存储多少个经验就可以进行训练了
            batch_size (int): DNN每次训练的样本大小
            memory_size (int): 最多能够同时存储的经验数
        '''
        self.net = net  # the size of the DNN
        self.training_interval = training_interval  # learn every #training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size

        # store all binary actions
        self.enumerate_actions = []

        # stored # memory entry
        self.memory_counter = 1

        # store training cost
        self.cost_his = []

        # initialize zero memory [h, m]
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))

        # construct memory network
        self._build_net()

    def _build_net(self):
        '''
        使用keras构建网络，keras的backen是tensorflow
        '''
        self.model = keras.Sequential([
            layers.Dense(self.net[1], activation='relu', input_dim=self.net[0]),  # the first hidden layer
            layers.Dense(self.net[2], activation='relu'),  # the second hidden layer
            layers.Dense(self.net[-1], activation='sigmoid')  # the output layer
        ])
        self.model.compile(optimizer=keras.optimizers.Adam(lr=self.lr), loss=tf.losses.binary_crossentropy,
                           metrics=['accuracy'])

    def remember(self, h, m):
        '''
        Args:
            h:信道增益
            m:
        存储经验，当经验池中的经验数量大于经验池最大能够存储的数量就把最旧的一个给覆盖掉
        '''
        # replace the old memory with new memory
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, m))

        self.memory_counter += 1

    def encode(self, h, m):
        # encoding the entry
        self.remember(h, m)
        # train the DNN every 10 step
        # if self.memory_counter> self.memory_size / 2 and self.memory_counter % self.training_interval == 0:
        # 每存储了self.training_interval个经验之后，就可以进行训练了
        if self.memory_counter % self.training_interval == 0:
            self.learn()

    def learn(self):
        # sample batch memory from all memory
        # 从所有的经验中随机的取出 batch_size个经验，然后进行训练
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        h_train = batch_memory[:, 0: self.net[0]]
        m_train = batch_memory[:, self.net[0]:]

        # print(h_train)          # (128, 10)
        # print(m_train)          # (128, 10)

        # train the DNN
        # 开始进行训练
        hist = self.model.fit(h_train, m_train, verbose=0)
        self.cost = hist.history['loss'][0]
        assert (self.cost > 0)
        self.cost_his.append(self.cost)

    def decode(self, h, k=1, mode='OP'):

        # to have batch dimension when feed into tf placeholder
        # 使用DNN预测  输入是信道增益 输出是[0,1]之间的float

        # 输入数据的形状是 (N,)
        # 神经网络接受的数据的形状是 (None,N)
        h = h[np.newaxis, :]

        # 使用神经网络进行预测，输出的是relaxed action
        m_pred = self.model.predict(h)

        # 使用相应的算法对神经网络预测出来的结果进行 量化 量化到0 or 1
        if mode == 'OP':
            # 根据OP算法 最终输出的是K个长度为N的数组，并且数组的元素是0或者1
            return self.knm(m_pred[0], k)
        elif mode == 'KNN':
            return self.knn(m_pred[0], k)
        else:
            print("The action selection must be 'OP' or 'KNN'")

    def knm(self, m, k=1):
        # 动态调整产生的的量化动作的数量
        # 产生K个量化的动作
        # return k order-preserving binary actions
        m_list = []
        # generate the ﬁrst binary ofﬂoading decision with respect to equation (8)
        m_list.append(1 * (m > 0.5))

        if k > 1:
            # generate the remaining K-1 binary ofﬂoading decisions with respect to equation (9)
            m_abs = abs(m - 0.5)
            idx_list = np.argsort(m_abs)[:k - 1]
            for i in range(k - 1):
                if m[idx_list[i]] > 0.5:
                    # set the \hat{x}_{t,(k-1)} to 0
                    m_list.append(1 * (m - m[idx_list[i]] > 0))
                else:
                    # set the \hat{x}_{t,(k-1)} to 1
                    m_list.append(1 * (m - m[idx_list[i]] >= 0))

        return m_list

    def knn(self, m, k=1):
        # list all 2^N binary offloading actions
        # 列出所有2^N中动作
        if len(self.enumerate_actions) == 0:
            import itertools
            self.enumerate_actions = np.array(list(map(list, itertools.product([0, 1], repeat=self.net[0]))))

        # the 2-norm
        sqd = ((self.enumerate_actions - m) ** 2).sum(1)
        idx = np.argsort(sqd)
        return self.enumerate_actions[idx[:k]]

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)) * self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.show()
