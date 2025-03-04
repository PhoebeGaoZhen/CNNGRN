# coding: utf-8
# 重复训练多次模型并绘制train_val_acc loss曲线
# 这里使用了expression+Y+生物学特征，样本是1*1709  1*1727
# 2022.01.06
# Gao Zhen

import warnings

warnings.filterwarnings('ignore')

import time
import os
import numpy as np
# import pandas as pd
import random
# import shutil
# from math import floor
import matplotlib.pyplot as plt
# from keras import models, layers
# from keras.utils import to_categorical  # 独热编码
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint  # 回调函数
from sklearn.model_selection import KFold, train_test_split
# from sklearn.preprocessing import LabelBinarizer  # 标签 二值化
# from sklearn.metrics import roc_curve, auc, roc_auc_score
# from scipy import interp
# import re
from CNNGRN import CNNGRN_definedFunc



# 重复训练多次模型并绘制train_val_acc loss曲线

pathts = ['D:\jupyter_project\CNNGRN\DATA\Ecoli\cold_time_3_replice.tsv',
          'D:\jupyter_project\CNNGRN\DATA\Ecoli\heat_time_3_replice.tsv',
          'D:\jupyter_project\CNNGRN\DATA\Ecoli\oxidativestress_time_3_replice.tsv']
pathY = 'D:\jupyter_project\CNNGRN\DATA\Ecoli\integrated_gold_network.tsv'  # 黄金网络的路径
pathTF = 'D:\jupyter_project\CNNGRN\DATA\Ecoli\TF_from_gold.tsv'  # TF的路径
# 读入原始数据 得到TF序列 和 靶基因序列
TFsID, genesID = CNNGRN_definedFunc.readRawData_Ecoli(pathTF, pathts[0])  # TFsID[0]:(array(['tdca'], dtype=object), 1),  genesID[0]:genesID[0]

# 将网络的 调控关联（pathY） 转换成 矩阵Y （geneNetwork）
geneNetwork = CNNGRN_definedFunc.createGRN_Ecoli(pathY, TFsID, genesID)

network = ['Ecoli_cold', 'Ecoli_heat', 'Ecoli_oxid']

for net in range(3):
    print(network[net] + '正在进行训练.............................................................')
    # 为每个网络构建样本
    pathts_net = pathts[net]
    positive_data, negative_data = CNNGRN_definedFunc.createSamples_Ecoli(pathts_net, pathTF,geneNetwork)

    # 4. 训练10次模型并绘制曲线
    # 重复训练10次，每次都打乱顺序
    all_acc = []
    all_val_acc = []
    all_loss = []
    all_val_loss = []
    for ki in range(10):
        print('\n')
        print(network[net])
        print("\n第{}次训练..........\n".format(ki + 1))

        # 打乱正样本和负样本
        random.shuffle(positive_data)
        random.shuffle(negative_data)
        # 随机选择与正样本数目相同的负样本并合并得到全部训练的样本alldata，并进行打乱
        alldata = np.vstack((positive_data, negative_data[0:3080]))  #
        random.shuffle(alldata)
        # 将全部样本进行转换得到样本 标签 坐标
        dataX, labelY, position = CNNGRN_definedFunc.transform_data(alldata)  # 获取样本和标签

        # 3. 创建模型
        if net == 2:
            hello = 1713
        else:
            hello = 1695
        model = CNNGRN_definedFunc.create_model_Ecoli(hello)

        # 3.1 增加回调函数
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

        # 3.2 划分数据集
        (trainXX, testXX, trainYY, testYY) = train_test_split(dataX, labelY, test_size=0.2, random_state=1,
                                                              shuffle=True)

        # 4. 训练模型
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
                                                                                               all_val_acc, all_loss,
                                                                                               all_val_loss)
    print('平均结果计算完成')

    # 绘制 acc loss曲线
    acc_name = 'Ecoli Training and validation accuracy ' + network[net] + '.png'
    loss_name = 'Ecoli Training and validation loss ' + network[net] + '.png'
    acc_title = 'Training and validation accuracy of Ecoli ' + network[net]
    loss_title = 'Training and validation loss of Ecoli ' + network[net]

    plt.figure()
    plt.plot(epochs, avg_acc, label='Training average accuracy')
    plt.plot(epochs, avg_val_acc, label='Validation average accuracy')
    plt.title(acc_title)
    plt.legend(loc='lower right')
    plt.savefig(acc_name, dpi=600)

    plt.figure()
    plt.plot(epochs, avg_loss, label='Training average loss')
    plt.plot(epochs, avg_val_loss, label='Validation average loss')
    plt.title(loss_title)
    plt.legend()
    plt.savefig(loss_name, dpi=600)
    # plt.show()

