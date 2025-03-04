# coding: utf-8
"""
CNNGRN用到的自定义的函数

2022/03/16
anthor: Gao Zhen

"""


import warnings
warnings.filterwarnings('ignore')
import re
import time
import os
import numpy as np
import pandas as pd
import random
import shutil
from math import floor
import matplotlib.pyplot as plt
from keras import models,layers,regularizers
from keras.utils import to_categorical  # 独热编码
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint  # 回调函数
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelBinarizer,MinMaxScaler     # 标签 二值化
# from sklearn.metrics import roc_curve,auc,roc_auc_score
# from scipy import interp
from sklearn.metrics import confusion_matrix,roc_auc_score,matthews_corrcoef,roc_curve,auc
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score,precision_recall_curve
from math import log2
# 获取特征和标签

def transform_data(train_data):
    """
    train_data的结构：[(sample,label,（坐标）)]
    return :
        trainX, labelY, position
    """
    feature = []  # 特征
    label_ = []   # 标签
    position = [] # 坐标
    for i in range(len(train_data)):  # 15*2
        feature.append(train_data[i][0])
        label_.append(train_data[i][1])
        position.append(train_data[i][2])

    feature = np.array(feature)
    label_ = np.array(label_)
    # print('label_:',label_)
    dataX = feature[:,np.newaxis,:,np.newaxis]  #  (30, 2, 115, 1)
    print(dataX.shape)

    # 对标签 独热编码  1 0 表示0， 0 1表示 1
    lb = LabelBinarizer()
    labelY_ = lb.fit(label_)  # 标签二值化
#     print(labelY_)
    labelY_2 = lb.transform(label_)
    labelY = to_categorical(labelY_2,2)    # 二值化标签 转换成 独热编码
    # print('labelY',labelY)

    position = np.array(position)
    return dataX, labelY, position

# 绘制acc_loss曲线
def average_acc_loss(all_acc, all_val_acc, all_loss, all_val_loss):
    num = []
    for j in range(len(all_acc)):
        num.append(len(all_acc[j]))
    maxnum = max(num)

    # all_acc  all_val_acc  all_loss 的个数应该是一样的
    for h in range(len(all_acc)):
        #         print(all_acc[h])
        all_acc[h] = list(all_acc[h] + [0] * (maxnum - len(all_acc[h])))
        all_val_acc[h] = list(all_val_acc[h] + [0] * (maxnum - len(all_val_acc[h])))
        all_loss[h] = list(all_loss[h] + [0] * (maxnum - len(all_loss[h])))
        all_val_loss[h] = list(all_val_loss[h] + [0] * (maxnum - len(all_val_loss[h])))
        #         print(all_acc[h])
        all_acc[h] = np.array(all_acc[h])
        all_val_acc[h] = np.array(all_val_acc[h])
        all_loss[h] = np.array(all_loss[h])
        all_val_loss[h] = np.array(all_val_loss[h])

    all_acc = np.array(all_acc)
    all_val_acc = np.array(all_val_acc)
    all_loss = np.array(all_loss)
    all_val_loss = np.array(all_val_loss)

    # 求出平均值
    avg_acc = []
    avg_val_acc = []
    avg_loss = []
    avg_val_loss = []
    for g in range(len(all_acc[0])):
        b_acc = [i[g] for i in all_acc]
        b_val_acc = [i[g] for i in all_val_acc]
        b_loss = [i[g] for i in all_loss]
        b_val_loss = [i[g] for i in all_val_loss]
        print(b_acc)
        changdu = 0
        for bb in range(len(b_acc)):
            if b_acc[bb] != 0:
                changdu += 1

        avg_acc_s = np.sum(b_acc) / changdu
        avg_val_acc_s = np.sum(b_val_acc) / changdu
        avg_loss_s = np.sum(b_loss) / changdu
        avg_val_loss_s = np.sum(b_val_loss) / changdu

        avg_acc.append(avg_acc_s)
        avg_val_acc.append(avg_val_acc_s)
        avg_loss.append(avg_loss_s)
        avg_val_loss.append(avg_val_loss_s)
    epochs = range(1, len(avg_acc) + 1)

    return epochs, avg_acc, avg_val_acc, avg_loss, avg_val_loss

def standard(rawdata):
    new_data1 = np.zeros((rawdata.shape[0],rawdata.shape[1]))
    for i in range(rawdata.shape[0]):
        for j in range(rawdata.shape[1]):
            new_data1[i][j] = log2(rawdata[i][j])
    Standard_data = MinMaxScaler().fit_transform(new_data1)
    return Standard_data

# 获得性能指标
def scores(y_test,y_pred,th=0.5):
    y_predlabel = [(0 if item<th else 1) for item in y_pred]
    tn,fp,fn,tp = confusion_matrix(y_test,y_predlabel).flatten()
    SPE = tn*1./(tn+fp)
    MCC = matthews_corrcoef(y_test,y_predlabel)
    Recall = recall_score(y_test, y_predlabel)
    Precision = precision_score(y_test, y_predlabel)
    F1 = f1_score(y_test, y_predlabel)
    Acc = accuracy_score(y_test, y_predlabel)
    AUC = roc_auc_score(y_test, y_pred)
    precision_aupr, recall_aupr, _ = precision_recall_curve(y_test, y_pred)
    AUPR = auc(recall_aupr, precision_aupr)
    return [Recall,SPE,Precision,F1,MCC,Acc,AUC,AUPR]

# 构建gene100 CNN模型
def create_model_gene100(input_shape):
    # 搭建网络 ,无正则化
    model = models.Sequential()

    model.add(layers.Conv2D(8, (1, 30), border_mode='same', activation="relu", input_shape=(1, input_shape, 1)))
    model.add(layers.MaxPooling2D((1, 10), padding='same'))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(16, (1, 30), border_mode='same', activation="relu"))
    model.add(layers.MaxPooling2D((1, 10), padding='same'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())

    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(2, activation="softmax"))

    # 配置优化设置
    model.compile(optimizer='rmsprop',
                  loss="binary_crossentropy",
                  metrics=["acc"])
    return model


# 读取gene100原始数据
def readRawData_gene100(path, pathts):
    '''
    path  调控网络（三列）的路径
    pathts  时间序列表达值的路径
    '''
    # 5个网络，每个网络的调控关联
    gene10 = pd.read_csv(path, sep='\t', header=None)  # 176*3
    gene10 = np.array(gene10)  # (176, 3)

    # 5个网络的时间序列表达值
    gene10_ts = pd.read_csv(pathts, sep='\t')  # (210, 100)
    gene10_ts = np.array(gene10_ts)  # （210,100）
    gene10_ts = np.transpose(gene10_ts)  # 转置 (100, 210)

    return gene10, gene10_ts


# 根据gene100原始调控网络构建GRN（Y）
def createGRN_gene100(gene10):
    rowNumber = []
    colNumber = []
    for i in range(len(gene10)):
        row = gene10[i][0]
        rownum = re.findall("\d+",row)
        rownumber = int(np.array(rownum))
        rowNumber.append(rownumber)

        col = gene10[i][1]
        colnum = re.findall("\d+",col)
        colnumber = int(np.array(colnum))
        colNumber.append(colnumber)

    geneNetwork = np.zeros((gene10.shape[0],gene10.shape[0]))
    for i in range(len(rowNumber)):
        r = rowNumber[i]-1
        c = colNumber[i]-1
        geneNetwork[r][c] = 1
#     print(np.sum(geneNetwork))
#     保存geneNetwork
#     data1 = pd.DataFrame(geneNetwork)
#     data1.to_csv('D:\jupyter_project\CNNGRN\DATA\DREAM100_samples\geneNetwork_100_'+str(net+1)+'.csv')
    return geneNetwork



# 根据gene100表达值和相互作用谱构建样本 （expression+Y 620 620 620 620 620）
# node_embedding_matrix 100*256
def createSamples_gene100(gene10_ts, geneNetwork):
    sample_10_pos = []
    sample_10_neg = []
    labels_pos = []
    labels_neg = []
    positive_1_position = []
    negative_0_position = []
    for i in range(100):
        for j in range(100):
            temp11 = gene10_ts[i]  # (210,)
            temp12 = geneNetwork[i]  # (256,)
            temp21 = gene10_ts[j]  # (210,)
            temp22 = geneNetwork[j]  # (256,)

            temp1 = np.hstack((temp11, temp12))  # (466,)  210+256
            temp2 = np.hstack((temp21, temp22))  # (456,)  210+256
            temp = np.hstack((temp1, temp2))  # 456+456=932
            # temp = np.hstack((temp11, temp21))  # (420,)
            # print('**************************temp1.shape')
            # print(temp1.shape)
            # print('**************************temp2.shape')
            # print(temp2.shape)
            # print('**************************temp.shape')
            # print(temp.shape)
            label = int(geneNetwork[i][j])

            if label == 1:
                sample_10_pos.append(temp)
                labels_pos.append(label)
                positive_1_position.append((i, j))


            else:
                sample_10_neg.append(temp)
                labels_neg.append(label)
                negative_0_position.append((i, j))

    # bind  feature (sample) and  label
    positive_data = list(zip(sample_10_pos, labels_pos, positive_1_position))
    negative_data = list(zip(sample_10_neg, labels_neg, negative_0_position))

    return positive_data, negative_data

# 读取Ecoli原始数据
def readRawData_Ecoli(pathTF, pathGene):
    '''
    pathTF  调控网络（三列）的路径
    pathGene  靶基因序列的路径
    '''
    # 转录因子
    TFs = pd.read_csv(pathTF, sep='\t', header=None)  # (163, 1)  TF[0][0] = 'tdca'
    TFs = np.array(TFs)  # (163, 1)
    # print(type(TFs))
    ID = [i for i in range(1, 163 + 1)]
    TFsID = list(zip(TFs, ID))  # (array(['tdca'], dtype=object), 1
    # TFs.shape
    # type(TFsID)

    # 基因
    cold_ts = pd.read_csv(pathGene, sep='\t')
    gene = cold_ts.columns  # (1484,)   gene[0]='nhaa'
    gene = np.array(gene)
    ID = [i for i in range(1, 1484 + 1)]
    genesID = list(zip(gene, ID))  # ('nhaa', 1),  1484 个
    # type(gene)
    # genesID

    return TFsID, genesID



# Ecoli geneNetwork  将调控关联 integrated_gold_network转换成 矩阵Y （geneNetwork）
def createGRN_Ecoli(pathY, TFsID, genesID):
    # 网络的调控关联
    integrated_gold_network = pd.read_csv(pathY, sep='\t', header=None)  # (3080, 3)
    integrated_gold_network = np.array(integrated_gold_network)
    # integrated_gold_network
    rowNumber = []
    colNumber = []
    for i in range(len(integrated_gold_network)):  # 3080
        tf = integrated_gold_network[i][0]
        for j in range(len(TFsID)):
            if tf == TFsID[j][0]:
                rownum = TFsID[j][1]
                #         print(rownum)
                rowNumber.append(rownum)  # 3080

        gene = integrated_gold_network[i][1]
        for k in range(len(genesID)):  # 1484
            if gene == genesID[k][0]:
                colnum = genesID[k][1]
                #         print(rownum)
                colNumber.append(colnum)  # 3080

    geneNetwork = np.zeros((163, 1484))
    for i in range(len(rowNumber)):
        r = rowNumber[i] - 1
        c = colNumber[i] - 1
        geneNetwork[r][c] = 1

    # data1 = pd.DataFrame(geneNetwork)
    #     data1.to_csv('D:\jupyter_project\CNNGRN\DATA\Ecoli_samples\geneNetwork_Ecoli.csv')
    #     geneNetwork  # np.sum(geneNetwork)  ==3080

    return geneNetwork



# 构建Ecoli样本（expression+Y 1695 1695 1713）
def createSamples_Ecoli(pathts, pathtf,geneNetwork):
    # 163个TF在1484中的位置已经找到，根据位置 取出对应的时间序列表达值，定义为新的矩阵
    cold_ts1 = pd.read_csv(pathts, sep='\t')
    gene = cold_ts1.columns  # (1484,)   gene[0]='nhaa'
    gene = np.array(gene)

    cold_ts = np.array(cold_ts1)  # (24, 1484)
    cold_ts = np.transpose(cold_ts)  # 转置 (1484, 24)
    cold_ts_stand = standard(cold_ts)

    TFs = pd.read_csv(pathtf, sep='\t', header=None)  # (163, 1)  TF[0][0] = 'tdca'
    TFs = np.array(TFs)  # (163, 1)

    hahaTF = TFs.flatten()
    genelist = gene.tolist()
    # 找到转录因子在1484中的位置
    Number = []
    jNumber = []
    for i in range(len(hahaTF)):
        name = hahaTF[i]
        for j in range(len(genelist)):
            if name == genelist[j]:
                number = genelist.index(name)
                Number.append(number)
                jNumber.append(j)
    TFexpression = []
    for i in range(len(Number)):
        TFexpression.append(cold_ts_stand[Number[i]])
    # TFexpression的TF顺序和geneNetwork 行中TF的顺序是否一样？？？？？？？？？？？？？？？？一样的

    # cold 网络可以产生163*1484个样本 包含基因表达值和相互作用谱（Y）
    sample_cold_pos = []
    sample_cold_neg = []
    labels_pos = []
    labels_neg = []
    positive_1_position = []
    negative_0_position = []
    # 这里是不对的  163个TF在1484中的位置已经找到，根据位置取出对应的时间序列表达值，定义为新的矩阵
    for i in range(163):
        for j in range(1484):
            # 转录因子的表达值
            tf1 = TFexpression[i]  # (24,)
            # 转录因子的相互作用谱
            tf2 = geneNetwork[i]  # (1484,)  维度太高，是否能降维？？？？？
            #         print(tf1.shape)
            #         print(tf2.shape)
            # 靶基因的表达值
            target1 = cold_ts_stand[j]  # (24,)
            # 靶基因的表达值
            target2 = geneNetwork[:, j]  # (163,)
            #         print(target1.shape)
            #         print(target2.shape)

            temp1 = np.hstack((tf1, tf2))  # (1508,)
            #         print(temp1.shape)
            temp2 = np.hstack((target1, target2))  # (187,)
            #         print(temp2.shape)
            temp = np.hstack((temp1, temp2))  # (1695)
            #             print(temp.shape)
            #         sample_cold.append(temp) # 241892个样本


            label = int(geneNetwork[i][j])
            if label == 1:
                sample_cold_pos.append(temp)
                labels_pos.append(label)
                positive_1_position.append((i, j))

            else:
                sample_cold_neg.append(temp)
                labels_neg.append(label)
                negative_0_position.append((i, j))
    # 将 feature (sample) 与 label 绑在一起
    positive_data = list(zip(sample_cold_pos, labels_pos, positive_1_position))  # len
    negative_data = list(zip(sample_cold_neg, labels_neg, negative_0_position))  # len
    #     samplePath1 = 'D:\jupyter_project\CNNGRN\DATA\Ecoli_samples\samples_Ecoli_cold_pos.npz'
    #     np.savez(samplePath1,sample=sample_cold_pos,label=labels_pos)
    #     samplePath2 = 'D:\jupyter_project\CNNGRN\DATA\Ecoli_samples\samples_Ecoli_cold_neg.npz'
    #     np.savez(samplePath2,sample=sample_cold_neg,label=labels_neg)
    #     print(len(labels_pos))
    #     print(len(labels_neg))

    return positive_data, negative_data



# 构建Ecoli样本（expression ）
def createSamples_Ecoli_2(pathts, pathtf,geneNetwork):
    print('createSamples_Ecoli_2(pathts, pathtf, geneNetwork, pathFeature)')
    # 163个TF在1484中的位置已经找到，根据位置取出对应的时间序列表达值，定义为新的矩阵
    cold_ts1 = pd.read_csv(pathts, sep='\t')
    gene = cold_ts1.columns  # (1484,)   gene[0]='nhaa'
    gene = np.array(gene)

    cold_ts = np.array(cold_ts1)  # (24, 1484)
    cold_ts = np.transpose(cold_ts)  # 转置 (1484, 24)

    TFs = pd.read_csv(pathtf, sep='\t', header=None)  # (163, 1)  TF[0][0] = 'tdca'
    TFs = np.array(TFs)  # (163, 1)

    hahaTF = TFs.flatten()
    genelist = gene.tolist()
    # 找到转录因子在1484中的位置
    Number = []
    jNumber = []
    for i in range(len(hahaTF)):
        name = hahaTF[i]
        for j in range(len(genelist)):
            if name == genelist[j]:
                number = genelist.index(name)
                Number.append(number)
                jNumber.append(j)
    TFexpression = []
    for i in range(len(Number)):
        TFexpression.append(cold_ts[Number[i]])
    # TFexpression的TF顺序和geneNetwork 行中TF的顺序是否一样？？？？？？？？？？？？？？？？一样的

    # cold 网络可以产生163*1484个样本 包含基因表达值和相互作用谱（Y）
    sample_cold_pos = []
    sample_cold_neg = []
    labels_pos = []
    labels_neg = []
    positive_1_position = []
    negative_0_position = []
    # 这里是不对的  163个TF在1484中的位置已经找到，根据位置取出对应的时间序列表达值，定义为新的矩阵
    for i in range(163):
        for j in range(1484):
            # 转录因子的表达值
            tf1 = TFexpression[i]  # (24,)
            # 靶基因的表达值
            target1 = cold_ts[j]  # (24,)
            temp = np.hstack((tf1, target1))  # (1695)

            label = int(geneNetwork[i][j])
            if label == 1:
                sample_cold_pos.append(temp)
                labels_pos.append(label)
                positive_1_position.append((i, j))

            else:
                sample_cold_neg.append(temp)
                labels_neg.append(label)
                negative_0_position.append((i, j))
    # 将 feature (sample) 与 label 绑在一起
    positive_data = list(zip(sample_cold_pos, labels_pos, positive_1_position))  # len
    negative_data = list(zip(sample_cold_neg, labels_neg, negative_0_position))  # len
    #     samplePath1 = 'D:\jupyter_project\CNNGRN\DATA\Ecoli_samples\samples_Ecoli_cold_pos.npz'
    #     np.savez(samplePath1,sample=sample_cold_pos,label=labels_pos)
    #     samplePath2 = 'D:\jupyter_project\CNNGRN\DATA\Ecoli_samples\samples_Ecoli_cold_neg.npz'
    #     np.savez(samplePath2,sample=sample_cold_neg,label=labels_neg)
    #     print(len(labels_pos))
    #     print(len(labels_neg))

    return positive_data, negative_data


# 构建Ecoli CNN模型
def create_model_Ecoli(hello):
    # 搭建网络 ,无正则化
    model = models.Sequential()
    model.add(layers.Conv2D(8, (1, 2), border_mode='same', activation="relu",
                            input_shape=(1, hello, 1)))  # cold heat 1695; oxid 1713
    # model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.MaxPooling2D((1, 1), padding='same'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(8, (1, 2), border_mode='same', activation="relu"))
    # model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.MaxPooling2D((1, 1), padding='same'))
    model.add(layers.Dropout(0.5))
    #     model.add(layers.Conv2D(128,(1,2),border_mode='same',activation="relu"))
    #     model.add(layers.MaxPooling2D((1, 1), padding='same'))

    model.add(layers.Flatten())

    model.add(layers.Dense(8, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(2, activation="softmax"))

    # 配置优化设置
    model.compile(optimizer='rmsprop',
                  loss="binary_crossentropy",
                  metrics=["acc"])
    return model



# 构建Ecoli样本（expression+特征 55 55 73）
def createSamples_Ecoli_3(pathts, pathtf, geneNetwork, pathFeature):
    # 读取特征并进行标准化
    # 读取1484个基因的所有信息 [ECKname,geneName,seq,geneName.1,length,GC,ATCG,ZCURVE_X,ZCURVE_Y,ZCURVE_Z,cumulativeSkew_GC,cumulativeSkew_AT]
    alldata = pd.read_csv(pathFeature, sep=',', header='infer')  # 1484*12
    # print(alldata.iloc[:,1])  # 数据的第几列
    # print(alldata[0:5] )      # 数据的第几行
    feature = alldata.iloc[:, 5:12]  # 7维序列特征  GC,ATCG,ZCURVE,cumulativeSkew
    feature = np.array(feature)
    feature_2 = MinMaxScaler().fit_transform(feature)

    # 163个TF在1484中的位置已经找到，根据位置取出对应的时间序列表达值，定义为新的矩阵
    cold_ts1 = pd.read_csv(pathts, sep='\t')
    gene = cold_ts1.columns  # (1484,)   gene[0]='nhaa'
    gene = np.array(gene)

    cold_ts = np.array(cold_ts1)  # (24, 1484)
    cold_ts = np.transpose(cold_ts)  # 转置 (1484, 24)

    TFs = pd.read_csv(pathtf, sep='\t', header=None)  # (163, 1)  TF[0][0] = 'tdca'
    TFs = np.array(TFs)  # (163, 1)

    hahaTF = TFs.flatten()
    genelist = gene.tolist()
    # 找到转录因子在1484中的位置
    Number = []
    jNumber = []
    for i in range(len(hahaTF)):
        name = hahaTF[i]
        for j in range(len(genelist)):
            if name == genelist[j]:
                number = genelist.index(name)
                Number.append(number)
                jNumber.append(j)
    TFexpression = []
    TFfeature = []
    for i in range(len(Number)):
        TFexpression.append(cold_ts[Number[i]])
        TFfeature.append(feature_2[Number[i]])
        # TFexpression的TF顺序和geneNetwork 行中TF的顺序是否一样？？？？？？？？？？？？？？？？一样的
    #     print(TFexpression)
    #     print(TFfeature)
    # cold 网络可以产生163*1484个样本 包含基因表达值和相互作用谱（Y）
    sample_cold_pos = []
    sample_cold_neg = []
    labels_pos = []
    labels_neg = []
    positive_1_position = []
    negative_0_position = []
    # 这里是不对的  163个TF在1484中的位置已经找到，根据位置取出对应的时间序列表达值，定义为新的矩阵
    for i in range(163):
        for j in range(1484):
            # 转录因子的表达值
            tf1 = TFexpression[i]  # (24,)
            # 转录因子的相互作用谱
            tf2 = TFfeature[i]  # (7,)
            #         print(tf1.shape)
            #             print(tf2.shape)
            # 靶基因的表达值
            target1 = cold_ts[j]  # (24,)
            # 靶基因的表达值
            target2 = feature_2[j]  # (7,)
            #         print(target1.shape)
            #         print(target2.shape)

            temp1 = np.hstack((tf1, tf2))  # (31,)
            #         print(temp1.shape)
            temp2 = np.hstack((target1, target2))  # (31,)
            #         print(temp2.shape)
            temp = np.hstack((temp1, temp2))  # (62)
            # print(temp.shape)
    #         sample_cold.append(temp) # 241892个样本


            label = int(geneNetwork[i][j])
            if label == 1:
                sample_cold_pos.append(temp)
                labels_pos.append(label)
                positive_1_position.append((i, j))

            else:
                sample_cold_neg.append(temp)
                labels_neg.append(label)
                negative_0_position.append((i, j))
    # 将 feature (sample) 与 label 绑在一起
    positive_data = list(zip(sample_cold_pos, labels_pos, positive_1_position))  # len
    negative_data = list(zip(sample_cold_neg, labels_neg, negative_0_position))  # len
    #     samplePath1 = 'D:\jupyter_project\CNNGRN\DATA\Ecoli_samples\samples_Ecoli_cold_pos.npz'
    #     np.savez(samplePath1,sample=sample_cold_pos,label=labels_pos)
    #     samplePath2 = 'D:\jupyter_project\CNNGRN\DATA\Ecoli_samples\samples_Ecoli_cold_neg.npz'
    #     np.savez(samplePath2,sample=sample_cold_neg,label=labels_neg)
    #     print(len(labels_pos))
    #     print(len(labels_neg))

    return positive_data, negative_data


# 构建Ecoli样本（expression+Y+特征 ）
def createSamples_Ecoli_4(pathts, pathtf, geneNetwork, pathFeature):

    # 读取特征并进行标准化
    # 读取1484个基因的所有信息 [ECKname,geneName,seq,geneName.1,length,GC,ATCG,ZCURVE_X,ZCURVE_Y,ZCURVE_Z,cumulativeSkew_GC,cumulativeSkew_AT]
    alldata = pd.read_csv(pathFeature, sep=',', header='infer')  # 1484*12
    # print(alldata.iloc[:,1])  # 数据的第几列
    # print(alldata[0:5] )      # 数据的第几行
    feature = alldata.iloc[:, 5:12]  # 7维序列特征  GC,ATCG,ZCURVE,cumulativeSkew
    feature = np.array(feature)
    feature_2 = MinMaxScaler().fit_transform(feature)

    # 163个TF在1484中的位置已经找到，根据位置取出对应的时间序列表达值，定义为新的矩阵
    cold_ts1 = pd.read_csv(pathts, sep='\t')
    gene = cold_ts1.columns  # (1484,)   gene[0]='nhaa'
    gene = np.array(gene)

    cold_ts = np.array(cold_ts1)  # (24, 1484)
    cold_ts = np.transpose(cold_ts)  # 转置 (1484, 24)

    TFs = pd.read_csv(pathtf, sep='\t', header=None)  # (163, 1)  TF[0][0] = 'tdca'
    TFs = np.array(TFs)  # (163, 1)

    hahaTF = TFs.flatten()
    genelist = gene.tolist()
    # 找到转录因子在1484中的位置
    Number = []
    jNumber = []
    for i in range(len(hahaTF)):
        name = hahaTF[i]
        for j in range(len(genelist)):
            if name == genelist[j]:
                number = genelist.index(name)
                Number.append(number)
                jNumber.append(j)
    TFexpression = []
    TFfeature = []
    for i in range(len(Number)):
        TFexpression.append(cold_ts[Number[i]])
        TFfeature.append(feature_2[Number[i]])
        # TFexpression的TF顺序和geneNetwork 行中TF的顺序是否一样？？？？？？？？？？？？？？？？一样的
    #     print(TFexpression)
    #     print(TFfeature)
    # cold 网络可以产生163*1484个样本 包含基因表达值和相互作用谱（Y）
    sample_cold_pos = []
    sample_cold_neg = []
    labels_pos = []
    labels_neg = []
    positive_1_position = []
    negative_0_position = []
    # 这里是不对的  163个TF在1484中的位置已经找到，根据位置取出对应的时间序列表达值，定义为新的矩阵
    for i in range(163):
        for j in range(1484):
            # 转录因子的表达值
            tf1 = TFexpression[i]  # (24,)
            # 转录因子的相互作用谱
            tf2 = TFfeature[i]  # (7,)
            tf3 = geneNetwork[i]  # (1484,)

            #         print(tf1.shape)
            #             print(tf2.shape)
            # 靶基因的表达值
            target1 = cold_ts[j]  # (24,)
            # 靶基因的表达值
            target2 = feature_2[j]  # (7,)
            target3 = geneNetwork[:, j]  # (163,)
            #         print(target1.shape)
            #         print(target2.shape)

            temp11 = np.hstack((tf1, tf2))  # (31,)
            temp1 = np.hstack((temp11, tf3))  #
            #         print(temp1.shape)
            temp21 = np.hstack((target1, target2))  # (31,)
            temp2 = np.hstack((temp21, target3))  #
            #         print(temp2.shape)
            temp = np.hstack((temp1, temp2))  # (1709)
            # print(temp.shape)
            #         sample_cold.append(temp) # 241892个样本


            label = int(geneNetwork[i][j])
            if label == 1:
                sample_cold_pos.append(temp)
                labels_pos.append(label)
                positive_1_position.append((i, j))

            else:
                sample_cold_neg.append(temp)
                labels_neg.append(label)
                negative_0_position.append((i, j))
    # 将 feature (sample) 与 label 绑在一起
    positive_data = list(zip(sample_cold_pos, labels_pos, positive_1_position))  # len
    negative_data = list(zip(sample_cold_neg, labels_neg, negative_0_position))  # len
    #     samplePath1 = 'D:\jupyter_project\CNNGRN\DATA\Ecoli_samples\samples_Ecoli_cold_pos.npz'
    #     np.savez(samplePath1,sample=sample_cold_pos,label=labels_pos)
    #     samplePath2 = 'D:\jupyter_project\CNNGRN\DATA\Ecoli_samples\samples_Ecoli_cold_neg.npz'
    #     np.savez(samplePath2,sample=sample_cold_neg,label=labels_neg)
    #     print(len(labels_pos))
    #     print(len(labels_neg))

    return positive_data, negative_data



def getDATA_DREAM4(networkID):
    negNum = [176, 249, 195, 211, 193]  # 负样本数量
    network = ['Gene100_NET1', 'Gene100_NET2', 'Gene100_NET3', 'Gene100_NET4', 'Gene100_NET5']
    print(network[networkID] + '正在计算中............................................................')

    # 1. 读取并转换数据格式
    #     gene10 = 'gene10_'+str(net+1)
    #     gene10_ts =  'gene10_'+str(net+1)+'_ts'
    path = 'D:\jupyter_project\CNNGRN\DATA\DREAM100\DREAM4_GoldStandard_InSilico_Size100_' + str(networkID + 1) + '.tsv'
    pathts = 'D:\jupyter_project\CNNGRN\DATA\DREAM100\insilico_size100_' + str(networkID + 1) + '_timeseries.tsv'
    gene10, gene10_ts = readRawData_gene100(path, pathts)
    geneNetwork = createGRN_gene100(gene10)

    # 3. 构建样本,得到所有正样本和所有负样本
    positive_data, negative_data = createSamples_gene100(gene10_ts, geneNetwork)

    # 4. 训练10次模型并绘制曲线
    # 重复训练10次，每次都打乱顺序
    random.shuffle(positive_data)
    random.shuffle(negative_data)
    # 随机选择与正样本数目相同的负样本并合并得到全部训练的样本alldata
    num = negNum[networkID]
    print('负样本数量为：' + str(num))
    # 分别是(352, 2) (498, 2) (390, 2) (422, 2) (386, 2)个(feature,label)
    alldata = np.vstack((positive_data, negative_data[0:num]))
    random.shuffle(alldata)
    # 将全部样本转换得到样本 标签 坐标
    dataX, labelY, position = transform_data(alldata)  # 获取样本和标签

    return dataX, labelY, position

def getDATA_DREAM4_all(networkID):
    negNum = [176, 249, 195, 211, 193]  # 负样本数量
    network = ['Gene100_NET1', 'Gene100_NET2', 'Gene100_NET3', 'Gene100_NET4', 'Gene100_NET5']
    print(network[networkID] + '正在计算中............................................................')

    # 1. 读取并转换数据格式
    #     gene10 = 'gene10_'+str(net+1)
    #     gene10_ts =  'gene10_'+str(net+1)+'_ts'
    path = 'D:\jupyter_project\CNNGRN\DATA\DREAM100\DREAM4_GoldStandard_InSilico_Size100_' + str(networkID + 1) + '.tsv'
    pathts = 'D:\jupyter_project\CNNGRN\DATA\DREAM100\insilico_size100_' + str(networkID + 1) + '_timeseries.tsv'
    gene10, gene10_ts = readRawData_gene100(path, pathts)
    geneNetwork = createGRN_gene100(gene10)

    # 3. 构建样本,得到所有正样本和所有负样本
    positive_data, negative_data = createSamples_gene100(gene10_ts, geneNetwork)

    # 4. 训练10次模型并绘制曲线
    # 重复训练10次，每次都打乱顺序
    random.shuffle(positive_data)
    random.shuffle(negative_data)
    # 随机选择与正样本数目相同的负样本并合并得到全部训练的样本alldata
    # num = negNum[networkID]
    # print('负样本数量为：' + str(num))
    # 直接返回所有正负样本，不再选取部分负样本！！！！！！！！！！！
    alldata = np.vstack((positive_data, negative_data))
    random.shuffle(alldata)
    # 将全部样本转换得到样本 标签 坐标
    dataX, labelY, position = transform_data(alldata)  # 获取样本和标签

    return dataX, labelY, position

def getDATA_Ecoli(networkID):
    network = ['cold', 'heat', 'oxid']
    print(network[networkID] + '正在计算中............................................................')

    # 1. 读取并转换数据格式
    #     gene10 = 'gene10_'+str(net+1)
    #     gene10_ts =  'gene10_'+str(net+1)+'_ts'
    # path = 'D:\jupyter_project\CNNGRN\DATA\DREAM100\DREAM4_GoldStandard_InSilico_Size100_' + str(networkID + 1) + '.tsv'
    # pathts = 'D:\jupyter_project\CNNGRN\DATA\DREAM100\insilico_size100_' + str(networkID + 1) + '_timeseries.tsv'

    pathts = ['D:\jupyter_project\CNNGRN\DATA\Ecoli\cold_time_3_replice.tsv',
              'D:\jupyter_project\CNNGRN\DATA\Ecoli\heat_time_3_replice.tsv',
              'D:\jupyter_project\CNNGRN\DATA\Ecoli\oxidativestress_time_3_replice.tsv']
    pathY = 'D:\jupyter_project\CNNGRN\DATA\Ecoli\integrated_gold_network.tsv'  # 黄金网络的路径
    pathTF = 'D:\jupyter_project\CNNGRN\DATA\Ecoli\TF_from_gold.tsv'  # TF的路径
    TFsID, genesID = readRawData_Ecoli(pathTF, pathts[0])
    # 将网络的 调控关联（pathY） 转换成 矩阵Y （geneNetwork）
    geneNetwork = createGRN_Ecoli(pathY, TFsID, genesID)

    # 3. 构建样本,得到所有正样本和所有负样本
    pathts_net = pathts[networkID]
    positive_data, negative_data = createSamples_Ecoli(pathts_net, pathTF, geneNetwork)
    # positive_data, negative_data = createSamples_gene100(gene10_ts, geneNetwork)

    # 4. 训练10次模型并绘制曲线
    # 重复训练10次，每次都打乱顺序
    random.shuffle(positive_data)
    random.shuffle(negative_data)
    # 随机选择与正样本数目相同的负样本并合并得到全部训练的样本alldata
    # num = 3080
    print('负样本数量为：' + str(3080))
    alldata = np.vstack((positive_data, negative_data[0:3080]))
    random.shuffle(alldata)
    # 将全部样本转换得到样本 标签 坐标
    dataX, labelY, position = transform_data(alldata)  # 获取样本和标签

    return dataX, labelY, position

def getDATA_Ecoli_all(networkID):
    network = ['cold', 'heat', 'oxid']
    print(network[networkID] + '正在计算中............................................................')

    # 1. 读取并转换数据格式
    #     gene10 = 'gene10_'+str(net+1)
    #     gene10_ts =  'gene10_'+str(net+1)+'_ts'
    # path = 'D:\jupyter_project\CNNGRN\DATA\DREAM100\DREAM4_GoldStandard_InSilico_Size100_' + str(networkID + 1) + '.tsv'
    # pathts = 'D:\jupyter_project\CNNGRN\DATA\DREAM100\insilico_size100_' + str(networkID + 1) + '_timeseries.tsv'

    pathts = ['D:\jupyter_project\CNNGRN\DATA\Ecoli\cold_time_3_replice.tsv',
              'D:\jupyter_project\CNNGRN\DATA\Ecoli\heat_time_3_replice.tsv',
              'D:\jupyter_project\CNNGRN\DATA\Ecoli\oxidativestress_time_3_replice.tsv']
    pathY = 'D:\jupyter_project\CNNGRN\DATA\Ecoli\integrated_gold_network.tsv'  # 黄金网络的路径
    pathTF = 'D:\jupyter_project\CNNGRN\DATA\Ecoli\TF_from_gold.tsv'  # TF的路径
    TFsID, genesID = readRawData_Ecoli(pathTF, pathts[0])
    # 将网络的 调控关联（pathY） 转换成 矩阵Y （geneNetwork）
    geneNetwork = createGRN_Ecoli(pathY, TFsID, genesID)

    # 3. 构建样本,得到所有正样本和所有负样本
    pathts_net = pathts[networkID]
    positive_data, negative_data = createSamples_Ecoli(pathts_net, pathTF, geneNetwork)
    # positive_data, negative_data = createSamples_gene100(gene10_ts, geneNetwork)

    # 4. 训练10次模型并绘制曲线
    # 重复训练10次，每次都打乱顺序
    random.shuffle(positive_data)
    random.shuffle(negative_data)
    # 随机选择与正样本数目相同的负样本并合并得到全部训练的样本alldata
    # num = 3080
    # print('负样本数量为：' + str(3080))
    alldata = np.vstack((positive_data, negative_data))
    random.shuffle(alldata)
    # 将全部样本转换得到样本 标签 坐标
    dataX, labelY, position = transform_data(alldata)  # 获取样本和标签

    return dataX, labelY, position

def test_model_DREAM4(model, networkID, cishu):
    dataX, labelY, position = getDATA_DREAM4(networkID)  # network1
    # 利用调用的模型预测新数据
    aucROC_all = []
    auPR_all = []
    for i in range(cishu):
        score_1 = model.predict(dataX)
        Recall1, SPE1, Precision1, F11, MCC1, Acc1, aucROC1, AUPR1 = scores(labelY[:, 1], score_1[:, 1],
                                                                                           th=0.5)
        aucROC_all.append(aucROC1)
        auPR_all.append(AUPR1)

    aucROC_mean = np.mean(aucROC_all)
    aucROC_mean = float('{:.4f}'.format(aucROC_mean))
    aucROC_std = np.std(aucROC_all)
    aucROC_std = float('{:.4f}'.format(aucROC_std))

    aucPR_mean = np.mean(auPR_all)
    aucPR_mean = float('{:.4f}'.format(aucPR_mean))
    # print(aucROC_all)
    # print("The aucROC_mean of network "+ str(networkID + 1)+" is " + str(aucROC_mean))
    # print("The aucROC_std of network " + str(networkID + 1) + " is " + str(aucROC_std))
    return aucROC_mean,aucPR_mean
def test_model_Ecoli(model, networkID, cishu):
    dataX, labelY, position = getDATA_Ecoli(networkID)  # network1
    # 利用调用的模型预测新数据
    aucROC_all = []
    auPR_all = []
    for i in range(cishu):
        score_1 = model.predict(dataX)
        Recall1, SPE1, Precision1, F11, MCC1, Acc1, aucROC1, AUPR1 = scores(labelY[:, 1], score_1[:, 1],
                                                                                           th=0.5)
        aucROC_all.append(aucROC1)
        auPR_all.append(AUPR1)

    aucROC_mean = np.mean(aucROC_all)
    aucROC_mean = float('{:.4f}'.format(aucROC_mean))
    aucROC_std = np.std(aucROC_all)
    aucROC_std = float('{:.4f}'.format(aucROC_std))

    aucPR_mean = np.mean(auPR_all)
    aucPR_mean = float('{:.4f}'.format(aucPR_mean))
    # print(aucROC_all)
    print("The aucROC_mean of network "+ str(networkID + 1)+" is " + str(aucROC_mean))
    print("The aucPR_mean of network " + str(networkID + 1) + " is " + str(aucPR_mean))
    return aucROC_mean,aucPR_mean