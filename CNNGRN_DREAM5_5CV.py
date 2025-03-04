# coding: utf-8
"""
样本：expression+Y，维度是1*620，但是没有标签泄露，去掉了
模型：conv+maxpooling+dropout(0.3)+BN(0.8)+conv+maxpooling+dropout(0.5)+BN(0.8)+FC+FC
10次五折交叉验证，平均AUC值，标准差和方差, AUPR F1等指标
2022/01/06
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
# import matplotlib.pyplot as plt
# from keras import models,layers,regularizers
# from keras.utils import to_categorical  # 独热编码
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint  # 回调函数
from sklearn.model_selection import KFold, train_test_split
# from sklearn.preprocessing import LabelBinarizer    # 标签 二值化
from sklearn.metrics import roc_curve,auc,roc_auc_score
# from scipy import interp
from CNNGRN import CNNGRN_definedFunc


negNum = [4012, 518, 2066, 3940]  # 负样本数量
network = ['DREAM5_NET1', 'DREAM5_NET2', 'DREAM5_NET3', 'DREAM5_NET4']
network_dict = {"10次5CV的AUROC平均值为":0,
                "10次5CV的AUROC标准差为":0,
                "10次5CV的AUPR平均值为": 0,
                "10次5CV的AUPR标准差为": 0,
                "10次5CV的Recall平均值为": 0,
                "10次5CV的Recall标准差为": 0,
                "10次5CV的SPE平均值为": 0,
                "10次5CV的SPE标准差为": 0,
                "10次5CV的Precision平均值为": 0,
                "10次5CV的Precision标准差为": 0,
                "10次5CV的F1平均值为": 0,
                "10次5CV的F1标准差为": 0,
                "10次5CV的MCC平均值为": 0,
                "10次5CV的MCC标准差为": 0,
                "10次5CV的Acc平均值为": 0,
                "10次5CV的Acc标准差为": 0}

# for net in range(4):  # 这里只考虑第一个网络
net = 3 # 第一个网络
print(network[net] + '正在训练中...................................................................')
# 1. 读取并转换数据格式
#     gene10 = 'gene10_'+str(net+1)
#     gene10_ts =  'gene10_'+str(net+1)+'_ts'

path = 'D:\jupyter_project\CNNGRN\DATA\DREAM5\\DREAM5_NetworkInference_GoldStandard_Network' + str(net + 1) + '.tsv'
pathts = 'D:\jupyter_project\CNNGRN\DATA\DREAM5\\net' + str(net + 1) + '_expression_data.tsv'
gene10, gene10_ts = CNNGRN_definedFunc.readRawData_gene100(path, pathts)
dim_ts = gene10_ts.shape[1]
# 2. 将第一个网络的 调控关联（gene100_1） 转换成 矩阵Y （geneNetwork）
geneNetwork = CNNGRN_definedFunc.createGRN_gene100(gene10,gene10_ts.shape[0])
dim_GRN = geneNetwork.shape[0]
# 3. 构建样本,得到所有正样本和所有负样本
positive_data, negative_data = CNNGRN_definedFunc.createSamples_gene100(gene10_ts, geneNetwork)

# 4. 10次5CV
kf = KFold(n_splits=5, shuffle=True)  # 初始化KFold
# AUROCs = []
# AUPRs = []
# Recalls = []
# SPEs = []
# Precisions = []
# F1s = []
# MCCs = []
# Accs = []
netavgAUROCs = []  # 存放一个网络10次5CV的平均AUC
netavgAUPRs = []
netavgRecalls = []
netavgSPEs = []
netavgPrecisions = []
netavgF1s = []
netavgMCCs = []
netavgAccs = []

for ki in range(5):
    print('\n')
    print(network[net])
    print("\n第{}次5折交叉验证..........\n".format(ki + 1))
    # 打乱正样本和负样本
    random.shuffle(positive_data)
    random.shuffle(negative_data)
    # 随机选择与正样本数目相同的负样本并合并得到全部训练的样本alldata，并进行打乱
    num = negNum[net]
    print('负样本数量为：' + str(num))
    alldata = np.vstack(
        (positive_data, negative_data[0:num]))  # 分别是(352, 2) (498, 2) (390, 2) (422, 2) (386, 2)个(feature,label)
    random.shuffle(alldata)
    # 将全部样本进行转换得到样本 标签 坐标
    dataX, labelY, position = CNNGRN_definedFunc.transform_data(alldata)  # 获取样本和标签
    input_shape = dataX.shape[2]

    # 5CV
    AUROCs = []
    AUPRs = []
    Recalls = []
    SPEs = []
    Precisions = []
    F1s = []
    MCCs = []
    Accs = []
    for train_index, test_index in kf.split(dataX, labelY):  # 调用split方法切分数据
        # 6.1 划分4:1训练集 测试集
        #             print('train_index:%s , test_index: %s ' %(train_index,test_index))
        trainX, testX = dataX[train_index], dataX[test_index]  # testX.shape (71, 1, 620, 1)
        trainY, testY = labelY[train_index], labelY[test_index]
        positionX, positionY = position[train_index], position[test_index]
        # 去掉测试集中的标签信息
        for i in range(len(test_index)):
            row = positionY[i][0]
            col = positionY[i][1]
            testX[i][0][dim_ts + col] = 0  # 将测试样本Y部分的位置置0
            testX[i][0][dim_ts+dim_ts+dim_GRN + row] = 0  # 将测试样本Y部分的位置置0
        # print(geneNetwork[row][col]) # 这三个值是一样的，证明找对了
        # print(testX[i][0][dim_ts+col])
        # print(testX[i][0][dim_ts+dim_ts+dim_GRN+row])
        # print('*************************')

        # 6.2 创建并训练模型
        model = CNNGRN_definedFunc.create_model_gene100(input_shape)

        # 增加回调函数
        logTime = time.strftime("%Y-%m-%d-%H%M%S", time.localtime())  # 运行程序的时间
        log_file_path = 'log\\log_' + logTime + '.csv'  # 实验日志的路径
        trained_models_path = 'trained_models\\'  # 实验模型的路径
        if (os.path.exists(trained_models_path) != 1):
            os.mkdir(trained_models_path)

        # model callbacks
        patience = 5  # 如果验证精度在 patience 轮内没有改善，就触发回调函数。
        early_stop = EarlyStopping('val_acc', 0.0001, patience=patience)  # val_acc 在 patience 轮内没有提升0.01，就会触发 早停机制
        reduce_lr = ReduceLROnPlateau('val_acc', factor=0.001, patience=int(patience / 2),
                                      verbose=1)  # 性能没有提升时，调整学习率
        csv_logger = CSVLogger(log_file_path, append=True)  # True:追加 ; False: 覆盖
        model_names = trained_models_path + logTime + '.{epoch:02d}-{acc:2f}.h5'
        model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', mode="max", verbose=1,
                                           save_best_only=True, save_weights_only=True)

        # callbacks = [model_checkpoint,csv_logger,early_stop,reduce_lr]
        callbacks = [csv_logger, early_stop, reduce_lr]

        # 划分数据集
        (trainXX, testXX, trainYY, testYY) = train_test_split(trainX, trainY, test_size=0.2, random_state=1,
                                                              shuffle=True)

        # 训练模型
        history = model.fit(trainXX, trainYY, validation_data=(testXX, testYY), batch_size=100, epochs=200,
                            callbacks=callbacks)

        # 7. 预测剩下的正样本数据[获取测试集的预测值]
        score_1 = model.predict(testX)

        # 8. 计算AUC值
        # 1个网络1折的AUC
        Recall, SPE, Precision, F1, MCC, Acc, aucROC, AUPR = CNNGRN_definedFunc.scores(testY[:, 1], score_1[:, 1], th=0.5)



        # 1个网络5折的AUC
        AUROCs.append(aucROC)
        AUPRs.append(AUPR)
        Recalls.append(Recall)
        SPEs.append(SPE)
        Precisions.append(Precision)
        F1s.append(F1)
        MCCs.append(MCC)
        Accs.append(Acc)

        print('一次五折交叉验证（1个网络5折的AUC）的AUC')
        print('\nAUROCs:')
        print(AUROCs)
        print('\n')
    # 一次五折交叉验证（1个网络5折的AUC）的平均AUC值
    avg_AUROC = np.mean(AUROCs)
    avg_AUPR = np.mean(AUPRs)
    avg_Recalls = np.mean(Recalls)
    avg_SPEs = np.mean(SPEs)
    avg_Precisions = np.mean(Precisions)
    avg_F1s = np.mean(F1s)
    avg_MCCs = np.mean(MCCs)
    avg_Accs = np.mean(Accs)

    # 10次5CV的AUC值，有10个值
    netavgAUROCs.append(avg_AUROC)  # 10个AUC值，长度为10
    netavgAUPRs.append(avg_AUPR)
    netavgRecalls.append(avg_Recalls)
    netavgSPEs.append(avg_SPEs)
    netavgPrecisions.append(avg_Precisions)
    netavgF1s.append(avg_F1s)
    netavgMCCs.append(avg_MCCs)
    netavgAccs.append(avg_Accs)
print(network[net] + '十次五折交叉验证的所有AUC值--------------------------------------------')
print(netavgAUROCs)

print(network[net] + '---------------------------------------------------------------------')
# 10次5CV的AUC平均值、标准差，有1个值
AUROC_mean = np.mean(netavgAUROCs)
AUROC_std = np.std(netavgAUROCs, ddof=1)
AUPR_mean = np.mean(netavgAUPRs)
AUPR_std = np.std(netavgAUPRs)
Recall_mean = np.mean(netavgRecalls)
Recall_std = np.std(netavgRecalls)
SPE_mean = np.mean(netavgSPEs)
SPE_std = np.std(netavgSPEs)
Precision_mean = np.mean(netavgPrecisions)
Precision_std = np.std(netavgPrecisions)
F1_mean = np.mean(netavgF1s)
F1_std = np.std(netavgF1s)
MCC_mean = np.mean(netavgMCCs)
MCC_std = np.std(netavgMCCs)
Acc_mean = np.mean(netavgAccs)
Acc_std = np.std(netavgAccs)

AUROC_mean = float('{:.4f}'.format(AUROC_mean))
AUROC_std = float('{:.4f}'.format(AUROC_std))
AUPR_mean = float('{:.4f}'.format(AUPR_mean))
AUPR_std = float('{:.4f}'.format(AUPR_std))
Recall_mean = float('{:.4f}'.format(Recall_mean))
Recall_std = float('{:.4f}'.format(Recall_std))
SPE_mean = float('{:.4f}'.format(SPE_mean))
SPE_std = float('{:.4f}'.format(SPE_std))
Precision_mean = float('{:.4f}'.format(Precision_mean))
Precision_std = float('{:.4f}'.format(Precision_std))
F1_mean = float('{:.4f}'.format(F1_mean))
F1_std = float('{:.4f}'.format(F1_std))
MCC_mean = float('{:.4f}'.format(MCC_mean))
MCC_std = float('{:.4f}'.format(MCC_std))
Acc_mean = float('{:.4f}'.format(Acc_mean))
Acc_std = float('{:.4f}'.format(Acc_std))

# print(network[net] + "10次5CV的AUC平均值为：%.4f" % AUROC_mean)
# print(network[net] + "10次5CV的AUC方差为：%.4f" % AUROC_var)
# print(network[net] + "10次5CV的AUC标准差为:%.4f" % AUROC_std)

# 将AUC的平均值标准差存到字典中，用于后续保存
network_dict["10次5CV的AUROC平均值为"] = AUROC_mean
network_dict["10次5CV的AUROC标准差为"] = AUROC_std
network_dict["10次5CV的AUPR平均值为"] = AUPR_mean
network_dict["10次5CV的AUPR标准差为"] = AUPR_std
network_dict["10次5CV的Recall平均值为"] = Recall_mean
network_dict["10次5CV的Recall标准差为"] = Recall_std
network_dict["10次5CV的SPE平均值为"] = SPE_mean
network_dict["10次5CV的SPE标准差为"] = SPE_std
network_dict["10次5CV的Precision平均值为"] = Precision_mean
network_dict["10次5CV的Precision标准差为"] = Precision_std
network_dict["10次5CV的F1平均值为"] = F1_mean
network_dict["10次5CV的F1标准差为"] = F1_std
network_dict["10次5CV的MCC平均值为"] = MCC_mean
network_dict["10次5CV的MCC标准差为"] = MCC_std
network_dict["10次5CV的Acc平均值为"] = Acc_mean
network_dict["10次5CV的Acc标准差为"] = Acc_std

# 保存一个网络10次5CV的AUC平均值、标准差
network_dict_name = network[net]
filename = open('D:\\pycharmProjects\\CNNGRN\\results\\dream5\\' + network_dict_name + '_v2.txt', 'w')
for k, v in network_dict.items():
    filename.write(k + ':' + str(v))
    filename.write('\n')
filename.close()







