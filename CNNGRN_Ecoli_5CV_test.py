"""
样本：expression+Y，维度是1*620
模型：两层CNN
循环10次训练模型
绘制平均 train_val 曲线

这里是用network1训练，把 network 2345 做独立测试集

2022/07/19
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
from CNNGRN import CNNGRN_definedFunc
import pickle
import joblib

allnet_AUROC = []
allnet_AUROC_mean = []  # 3个网络的平均AUC值
allnet_AUROC_std = []
allnet_AUROC_var = []

# 创建并训练模型
hello = 1695
modelgene = CNNGRN_definedFunc.create_model_Ecoli(hello)

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

# 数据集
dataXC, labelYC, positionC = CNNGRN_definedFunc.getDATA_Ecoli(0)  # network  COLD
dataXH, labelYH, positionH = CNNGRN_definedFunc.getDATA_Ecoli(1)  # network  HEAT
dataXO, labelYO, positionO = CNNGRN_definedFunc.getDATA_Ecoli(2)  # network  OXID

# 以network1作为训练集，其余作为独立测试集
(trainXX, testXX, trainYY, testYY) = train_test_split(dataXO, labelYO, test_size=0.2, random_state=1,
                                                      shuffle=True)
# 训练模型
history = modelgene.fit(trainXX, trainYY, validation_data=(testXX, testYY), batch_size=4, epochs=200,
                    callbacks=callbacks)

aucROC_mean_C,aucPR_mean_C = CNNGRN_definedFunc.test_model_Ecoli(modelgene, 0, 30)
aucROC_mean_H,aucPR_mean_H = CNNGRN_definedFunc.test_model_Ecoli(modelgene, 1, 30)
aucROC_mean_O,aucPR_mean_O = CNNGRN_definedFunc.test_model_Ecoli(modelgene, 2, 30)



aucROC_mean_all_Ecoli = [aucROC_mean_C,aucROC_mean_H,aucROC_mean_O]
aucPR_mean_all_Ecoli = [aucPR_mean_C,aucPR_mean_H,aucPR_mean_O]

print(aucROC_mean_all_Ecoli,aucPR_mean_all_Ecoli)
# print(aucROC_mean_all_Ecoli,aucROC_std_all_Ecoli)

# modelgene.summary()

# 保存模型
# pickle.dump(modelgene, open(r".\\trained_models\\CNNGRN0428.dat", 'wb'))
# # 调用模型
# model_new = pickle.load(open(r".\\trained_models\\CNNGRN0428.dat", "rb"))
# # 保存模型
# joblib.dump(modelgene, '.\\trained_models\\CNNGRN0428joblib.pkl')
# # 读取模型
# model_new = joblib.load('.\\trained_models\\CNNGRN0428joblib.pkl')


