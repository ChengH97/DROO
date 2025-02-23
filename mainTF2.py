#  #################################################################
#  Deep Reinforcement Learning for Online Ofﬂoading in Wireless Powered Mobile-Edge Computing Networks
#
#  This file contains the main code of DROO. It loads the training samples saved in ./data/data_#.mat, splits the samples into two parts (training and testing data constitutes 80% and 20%), trains the DNN with training and validation samples, and finally tests the DNN with test data.
#
#  Input: ./data/data_#.mat
#    Data samples are generated according to the CD method presented in [2]. There are 30,000 samples saved in each ./data/data_#.mat, where # is the user number. Each data sample includes
#  -----------------------------------------------------------------
#  |       wireless channel gain           |    input_h            |
#  -----------------------------------------------------------------
#  |       computing mode selection        |    output_mode        |
#  -----------------------------------------------------------------
#  |       energy broadcasting parameter   |    output_a           |
#  -----------------------------------------------------------------
#  |     transmit time of wireless device  |    output_tau         |
#  -----------------------------------------------------------------
#  |      weighted sum computation rate    |    output_obj         |
#  -----------------------------------------------------------------
#
#
#  References:
#  [1] 1. Liang Huang, Suzhi Bi, and Ying-Jun Angela Zhang, "Deep Reinforcement Learning for Online Offloading in Wireless Powered Mobile-Edge Computing Networks," in IEEE Transactions on Mobile Computing, early access, 2019, DOI:10.1109/TMC.2019.2928811.
#  [2] S. Bi and Y. J. Zhang, “Computation rate maximization for wireless powered mobile-edge computing with binary computation ofﬂoading,” IEEE Trans. Wireless Commun., vol. 17, no. 6, pp. 4177-4190, Jun. 2018.
#
# version 1.0 -- July 2018. Written by Liang Huang (lianghuang AT zjut.edu.cn)
#  #################################################################


import scipy.io as sio  # import scipy.io for .mat file I/
import numpy as np  # import numpy

# for tensorflow2
from memoryTF2 import MemoryDNN
from optimization import bisection

import time


def plot_rate(rate_his, rolling_intv=50):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    rate_array = np.asarray(rate_his)
    df = pd.DataFrame(rate_his)

    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15, 8))
    #    rolling_intv = 20

    plt.plot(np.arange(len(rate_array)) + 1, df.rolling(rolling_intv, min_periods=1).mean(), 'b')
    plt.fill_between(np.arange(len(rate_array)) + 1, df.rolling(rolling_intv, min_periods=1).min()[0],
                     df.rolling(rolling_intv, min_periods=1).max()[0], color='b', alpha=0.2)
    plt.ylabel('Normalized Computation Rate')
    plt.xlabel('Time Frames')
    plt.show()


def save_to_txt(rate_his, file_path):
    with open(file_path, 'w') as f:
        for rate in rate_his:
            f.write("%s \n" % rate)


if __name__ == "__main__":
    '''
        This algorithm generates K modes from DNN, and chooses with largest
        reward. The mode with largest reward is stored in the memory, which is
        further used to train the DNN.
        Adaptive K is implemented. K = max(K, K_his[-memory_size])
    '''

    N = 10  # number of users ,WDs的数量
    n = 300  # number of time frames ,一共模拟30000个时间片，以得到最终的数据    时间帧数，如果实际的任务数小于帧数，就循环的使用任务
    K = N  # initialize K = N ,初始化的K大小
    decoder_mode = 'OP'  # the quantization mode could be 'OP' (Order-preserving) or 'KNN'
    Memory = 1024  # capacity of memory structure 经验池的容量大小是1024条经验
    Delta = 32  # Update interval for adaptive K

    print(
        '#user = %d, #channel=%d, K=%d, decoder = %s, Memory = %d, Delta = %d' % (N, n, K, decoder_mode, Memory, Delta))
    # Load data
    # 从./data/data_10.mat中加载训练的数据
    channel = sio.loadmat('./data/data_%d' % N)['input_h']  # 信道增益的数据
    rate = sio.loadmat('./data/data_%d' % N)[  # 这个数据只是用在plot绘图上，没有用在训练上
        'output_obj']  # this rate is only used to plot figures; never used to train DROO.

    # increase h to close to 1 for better training; it is a trick widely adopted in deep learning
    channel = channel * 1000000  # 信道增益数据的量级在10^-6~10^-7量级之间，将数据向一附近靠拢减小浮点运算，为了更快的训练

    # generate the train and test data sample index
    # data are splitted as 80:20
    # training data are randomly sampled with duplication if n > total data size
    # 将数据按照8:2的比例进行划分
    split_idx = int(.8 * len(channel))
    # 前80%的数据是train数据 后20%的数据是test数据
    num_test = min(len(channel) - split_idx, n - int(.8 * n))  # training data size

    # 创建DNN 输入层有N个神经元 两个隐藏层 分别有120和80个神经元 输出层也有120个神经元
    mem = MemoryDNN(net=[N, 120, 80, N],
                    learning_rate=0.01,
                    training_interval=10,
                    batch_size=128,
                    memory_size=Memory
                    )

    start_time = time.time()

    rate_his = []
    rate_his_ratio = []
    mode_his = []
    k_idx_his = []
    K_his = []
    for i in range(n):  # 一共有30000个时间帧
        #############################################
        # 只是输出一下程序的执行进度
        # 用百分比表示
        if i % (n // 10) == 0:
            print("%0.1f" % (i / n))
        #############################################

        # 更新K 更新的算法见论文
        if i > 0 and i % Delta == 0:
            # index counts from 0
            if Delta > 1:
                max_k = max(k_idx_his[-Delta:-1]) + 1;
            else:
                max_k = k_idx_his[-1] + 1;
            K = min(max_k + 1, N)

        if i < n - num_test:
            # training  从训练集中取出数据
            i_idx = i % split_idx
        else:
            # test      从测试集中取出数据
            i_idx = i - n + num_test + split_idx

        h = channel[i_idx, :]  # 取出信道增益

        # the action selection must be either 'OP' or 'KNN'
        # 输入 (信道增益 K OP)
        # 输出 K个长度为N的数组，并且数组的元素是0或者1
        m_list = mem.decode(h, K, decoder_mode)

        ##################################################
        # 主要是这一块的内容没有弄得很清楚
        r_list = []
        for m in m_list:
            r_list.append(bisection(h / 1000000, m)[0])
        ##################################################

        # encode the mode with largest reward
        # 选出其中具有最大加权计算速率的动作
        mem.encode(h, m_list[np.argmax(r_list)])
        # the main code for DROO training ends here

        # the following codes store some interested metrics for illustrations
        # memorize the largest reward
        rate_his.append(np.max(r_list))
        rate_his_ratio.append(rate_his[-1] / rate[i_idx][0])
        # record the index of largest reward
        k_idx_his.append(np.argmax(r_list))
        # record K in case of adaptive K
        K_his.append(K)
        mode_his.append(m_list[np.argmax(r_list)])

    total_time = time.time() - start_time
    # 绘图
    # mem.plot_cost()
    # plot_rate(rate_his_ratio)

    print("Averaged normalized computation rate:", sum(rate_his_ratio[-num_test: -1]) / num_test)
    print('Total time consumed:%s' % total_time)
    print('Average time per channel:%s' % (total_time / n))

    # save data into txt
    save_to_txt(k_idx_his, "k_idx_his.txt")
    save_to_txt(K_his, "K_his.txt")
    save_to_txt(mem.cost_his, "cost_his.txt")
    save_to_txt(rate_his_ratio, "rate_his_ratio.txt")
    save_to_txt(mode_his, "mode_his.txt")
