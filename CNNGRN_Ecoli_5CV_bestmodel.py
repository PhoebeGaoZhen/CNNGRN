"""
这里是用network1训练，把 network 2345 做独立测试集
训练多次模型找到最好的模型

"""

import warnings
warnings.filterwarnings('ignore')
import time
import os
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint  # 回调函数
from sklearn.model_selection import KFold, train_test_split
from CNNGRN import CNNGRN_definedFunc
import pickle
import joblib


class CNNGRN():
    def __init__(self, dataX_train, labelY_train,dataX_test1, labelY_test1,dataX_test2, labelY_test2, number):
        self.dataX_train = dataX_train
        self.labelY_train = labelY_train
        self.dataX_test1 = dataX_test1
        self.labelY_test1 = labelY_test1
        self.dataX_test2= dataX_test2
        self.labelY_test2 = labelY_test2
        self.number = number
        # self.test_size = 0.2
        # self.batch_size = 4
        # self.epochs = 200

    def trainmodel(self):
        # seed = 7
        # np.random.seed(seed)

        # 创建并训练模型
        self.modelgene = CNNGRN_definedFunc.create_model_Ecoli(hello=1713)

        # 增加回调函数
        logTime = time.strftime("%Y-%m-%d-%H%M%S", time.localtime())  # 运行程序的时间
        log_file_path = 'log\\log_' + logTime + '.csv'  # 实验日志的路径
        trained_models_path = 'trained_models\\'  # 实验模型的路径
        if (os.path.exists(trained_models_path) != 1):
            os.mkdir(trained_models_path)

        # model callbacks
        patience = 10 # 如果验证精度在 patience 轮内没有改善，就触发回调函数。
        early_stop = EarlyStopping('val_acc', 0.0001, patience=patience)  # val_acc 在 patience 轮内没有提升0.01，就会触发 早停机制
        reduce_lr = ReduceLROnPlateau('val_acc', factor=0.001, patience=int(patience / 2), verbose=1)  # 性能没有提升时，调整学习率
        csv_logger = CSVLogger(log_file_path, append=True)  # True:追加 ; False: 覆盖
        # 使用第一个名字表示只要提升就会保存，使用第二个名称表示提升的模型会覆盖掉之前的模型
        model_names1 = trained_models_path+ str(self.number+1)  + logTime + '.{epoch:02d}-{acc:2f}.h5'
        model_names2 = trained_models_path + str(self.number+1) + '_CNNGRN_best_model_Ecoli_oxid.h5'
        model_checkpoint = ModelCheckpoint(model_names2, monitor='val_acc', mode="max", verbose=1, save_best_only=True,
                                           save_weights_only=True)

        callbacks = [model_checkpoint,csv_logger,early_stop,reduce_lr]
        # callbacks = [csv_logger, early_stop, reduce_lr]
        # 训练模型
        history = self.modelgene.fit(self.dataX_train, self.labelY_train, validation_split=0.2, batch_size=4, epochs=100, callbacks=callbacks)

        acc = history.history['acc']
        # print(acc)
        # print("train finished")
        score_train = self.modelgene.predict(self.dataX_train)
        # score_1 = self.modelgene.predict(self.dataX_test1)
        # score_2 = self.modelgene.predict(self.dataX_test2)

        Recallt, SPEt, Precisiont, F1t, MCCt, Acct, aucROCt, AUPRt = CNNGRN_definedFunc.scores(self.labelY_train[:, 1], score_train[:, 1],
                                                                            th=0.5)
        # Recall1, SPE1, Precision1, F11, MCC1, Acc1, aucROC1, AUPR1 = CNNGRN_definedFunc.scores(self.labelY_test1[:, 1],
        #                                                                                        score_1[:, 1],
        #                                                                                        th=0.5)
        # Recall2, SPE2, Precision2, F11, MCC2, Acc2, aucROC2, AUPR2 = CNNGRN_definedFunc.scores(self.labelY_test2[:, 1],
        #                                                                                        score_2[:, 1],
        #                                                                                        th=0.5)

        # allAUC = [aucROCt,aucROC1,aucROC2]
        allAUC = [aucROCt]
        allAUC2 = [float('{:.4f}'.format(i)) for i in allAUC]

        return allAUC2


if __name__ == '__main__':
    # 数据集
    dataX1, labelY1, position1 = CNNGRN_definedFunc.getDATA_Ecoli(0)  # cold network1
    dataX2, labelY2, position2 = CNNGRN_definedFunc.getDATA_Ecoli(1)  # heat network2
    dataX3, labelY3, position3 = CNNGRN_definedFunc.getDATA_Ecoli(2)  # oxid network3
    multi_allAUC = []
    multi_allAUC = np.array(multi_allAUC)
    # seed = 7
    # np.random.seed(seed)

    for i in range(10):
        # 以dataX1作为训练集，dataX2 dataX3 dataX4 dataX5作为独立测试集
        # modelgene = CNNGRN(dataX1, labelY1,dataX2, labelY2,dataX3, labelY3,dataX4, labelY4,dataX5, labelY5,i)
        # # 以dataX2作为训练集，其余作为测试集
        # modelgene = CNNGRN(dataX2, labelY2, dataX1, labelY1,  dataX3, labelY3, dataX4, labelY4, dataX5, labelY5,i)
        # # 以dataX3作为训练集，其余作为测试集
        # modelgene = CNNGRN(dataX3, labelY3, dataX1, labelY1, dataX2, labelY2, dataX4, labelY4, dataX5, labelY5,i)
        # # 以dataX4作为训练集，其余作为测试集
        # modelgene = CNNGRN(dataX4, labelY4, dataX1, labelY1, dataX2, labelY2, dataX3, labelY3, dataX5, labelY5,i)
        # # 以dataX5作为训练集，其余作为测试集
        modelgene = CNNGRN(dataX3, labelY3, dataX1, labelY1, dataX2, labelY2, i)

        allAUC_test = modelgene.trainmodel()
        allAUC_test = np.array(allAUC_test)
        # print('独立测试集结果为：')
        # print(allAUC_test)

        if i==0:
            multi_allAUC = allAUC_test
        else:
            multi_allAUC = np.vstack((multi_allAUC,allAUC_test))

    print(multi_allAUC.shape)
    print(multi_allAUC)
    pd.DataFrame(multi_allAUC).to_csv('net1_2_allAUC_save_0726.csv')

# modelgene.summary()
#
# # 保存模型
# pickle.dump(modelgene, open(r"D:\\pycharmProjects\\CNNGRN\\trained_models\\CNNGRN0429.dat", 'wb'))
# # 调用模型
# model_new = pickle.load(open("D:\\pycharmProjects\\CNNGRN\\trained_models\\CNNGRN0429.dat", "rb"))
# aucROC_mean_1,aucROC_std_1 = CNNGRN_definedFunc.test_model(model_new, 0, 30)

# # 保存模型
# joblib.dump(modelgene, '.\\trained_models\\CNNGRN0428joblib.pkl')
# # 读取模型
# model_new = joblib.load('.\\trained_models\\CNNGRN0428joblib.pkl')


