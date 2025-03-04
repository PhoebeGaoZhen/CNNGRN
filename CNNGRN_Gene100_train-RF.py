"""
样本：expression+Y，维度是1*620
模型：RF

2022/08/1
anthor: Gao Zhen
"""

import warnings
warnings.filterwarnings('ignore')
# import re
import time
import os
import numpy as np
# import pandas as pd
import random
# import shutil
# from math import floor
import matplotlib.pyplot as plt
# from keras import models,layers,regularizers
# from keras.utils import to_categorical  # 独热编码
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint  # 回调函数
from sklearn.model_selection import KFold, train_test_split
# from sklearn.preprocessing import LabelBinarizer    # 标签 二值化
from sklearn.metrics import roc_curve,auc,roc_auc_score
# from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from CNNGRN import CNNGRN_definedFunc
import pickle

def smooth_curve(points,factor = 0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

negNum = [176, 249, 195, 211, 193]  # 负样本数量
network = ['Gene100_NET1', 'Gene100_NET2', 'Gene100_NET3', 'Gene100_NET4', 'Gene100_NET5']
allnet_AUROC = []
allnet_AUROC_mean = []  # 3个网络的平均AUC值
allnet_AUROC_std = []
allnet_AUROC_var = []
for net in range(1):  # 这里只考虑第一个网络
    print(network[net] + '正在训练中............................................................')
    # 1. 读取并转换数据格式
    #     gene10 = 'gene10_'+str(net+1)
    #     gene10_ts =  'gene10_'+str(net+1)+'_ts'
    path = 'D:\jupyter_project\CNNGRN\DATA\DREAM100\DREAM4_GoldStandard_InSilico_Size100_' + str(net + 1) + '.tsv'
    pathts = 'D:\jupyter_project\CNNGRN\DATA\DREAM100\insilico_size100_' + str(net + 1) + '_timeseries.tsv'
    gene10, gene10_ts = CNNGRN_definedFunc.readRawData_gene100(path, pathts)

    '''
    2. 将第一个网络的 调控关联（gene100_1） 转换成 矩阵Y （geneNetwork）
    geneNetwork = CNNGRN_definedFunc.createGRN_gene100(gene10)
    这里不使用原始的geneNetwork，使用嵌入矩阵node_embedding_matrix
    '''
    geneNetwork = CNNGRN_definedFunc.createGRN_gene100(gene10)
    # 3. 构建样本,得到所有正样本和所有负样本
    positive_data, negative_data = CNNGRN_definedFunc.createSamples_gene100(gene10_ts, geneNetwork)

    # 4. 训练10次模型并绘制曲线
    # 重复训练10次，每次都打乱顺序
    all_acc = []
    all_val_acc = []
    all_loss = []
    all_val_loss = []
    for ki in range(2):
        print('\n')
        print(network[net])
        print("\n第{}次训练..........\n".format(ki + 1))

        random.shuffle(positive_data)
        random.shuffle(negative_data)
        # 随机选择与正样本数目相同的负样本并合并得到全部训练的样本alldata
        num = negNum[net]
        print('负样本数量为：' + str(num))
        # 分别是(352, 2) (498, 2) (390, 2) (422, 2) (386, 2)个(feature,label)
        alldata = np.vstack((positive_data, negative_data[0:num]))
        random.shuffle(alldata)
        # 将全部样本转换得到样本 标签 坐标
        dataX, labelY, position = CNNGRN_definedFunc.transform_data(alldata)  # 获取样本和标签

        # 6.2 创建并训练模型
        model = CNNGRN_definedFunc.create_model_gene100()

        # 增加回调函数
        logTime = time.strftime("%Y-%m-%d-%H%M%S", time.localtime())  # 运行程序的时间
        log_file_path = 'log\\log_' + logTime + '.csv'  # 实验日志的路径
        trained_models_path = 'trained_models\\'  # 实验模型的路径
        if (os.path.exists(trained_models_path) != 1):
            os.mkdir(trained_models_path)

        # model callbacks
        patience = 10  # 如果验证精度在 patience 轮内没有改善，就触发回调函数。
        early_stop = EarlyStopping('val_acc', 0.0001, patience=patience)  # val_acc 在 patience 轮内没有提升0.01，就会触发 早停机制
        reduce_lr = ReduceLROnPlateau('val_acc', factor=0.001, patience=int(patience / 2), verbose=1)  # 性能没有提升时，调整学习率
        csv_logger = CSVLogger(log_file_path, append=True)  # True:追加 ; False: 覆盖
        model_names = trained_models_path + logTime + '.{epoch:02d}-{acc:2f}.h5'
        model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', mode="max", verbose=1, save_best_only=True,
                                           save_weights_only=True)

        # callbacks = [model_checkpoint,csv_logger,early_stop,reduce_lr]
        callbacks = [csv_logger, early_stop, reduce_lr]

        # 划分数据集
        (trainXX, testXX, trainYY, testYY) = train_test_split(dataX, labelY, test_size=0.2, random_state=1,
                                                              shuffle=True)

        # 训练模型
        history = model.fit(trainXX, trainYY, validation_data=(testXX, testYY), batch_size=4, epochs=200,
                            callbacks=callbacks)

        model.summary()
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        all_acc.append(acc)
        all_val_acc.append(val_acc)
        all_loss.append(loss)
        all_val_loss.append(val_loss)
    # print(all_acc)
    epochs, avg_acc, avg_val_acc, avg_loss, avg_val_loss = CNNGRN_definedFunc.average_acc_loss(all_acc,
                                                                        all_val_acc, all_loss, all_val_loss)
    print('平均结果计算完成')

    # 绘制 acc loss曲线
    acc_name = 'GENE100 Training and validation accuracy ' + str(net + 1) + '2.png'
    loss_name = 'GENE100 Training and validation loss ' + str(net + 1) + '2.png'
    acc_title = 'Training and validation accuracy of GENE100 NET ' + str(net + 1)
    loss_title = 'Training and validation loss of GENE100 NET ' + str(net + 1)
    plt.figure()
    plt.plot(epochs, smooth_curve(avg_acc, factor=0.8), label='Training average accuracy')
    plt.plot(epochs, smooth_curve(avg_val_acc, factor=0.8), label='Validation average accuracy')
    # smooth_curve(avg_acc, factor=0.8)
    plt.title(acc_title)
    plt.legend(loc='lower right')
    plt.savefig(acc_name, dpi=600)
    plt.figure()

    plt.plot(epochs, smooth_curve(avg_loss, factor=0.8), label='Training average loss')
    plt.plot(epochs, smooth_curve(avg_val_loss, factor=0.8), label='Validation average loss')
    plt.title(loss_title)
    plt.legend(loc='lower left')
    plt.savefig(loss_name, dpi=600)
    # plt.show()




