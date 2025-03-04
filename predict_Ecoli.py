import warnings
warnings.filterwarnings('ignore')
import numpy as np
from CNNGRN import CNNGRN_definedFunc
import pandas as pd
seed= 7
np.random.seed(seed)

# 获取DREAM4 GENE100  network1 样本
# dataX1, labelY1, position1 = CNNGRN_definedFunc.getDATA_Ecoli_all(0)  # network cold
dataX2, labelY2, position2 = CNNGRN_definedFunc.getDATA_Ecoli_all(1)  # network heat
# dataX3, labelY3, position3 = CNNGRN_definedFunc.getDATA_Ecoli_all(2)  # network oxid


# create model
modelgene = CNNGRN_definedFunc.create_model_Ecoli(hello=1695)
# load weights 加载模型权重
modelgene.load_weights(r'D:\pycharmProjects\CNNGRN\results\CNNGRN_best_model_Ecoli_heat.h5')
modelgene.summary()
print('successfully loaded weights from hdf5 file')

# 预测
score_1 = modelgene.predict(dataX2)
# 将基因对和预测分数拼接在一起
pred_result = np.hstack((position2,score_1))

# 读取基因对的名称，将编号转换为名称
pathts = ['D:\jupyter_project\CNNGRN\DATA\Ecoli\cold_time_3_replice.tsv',
          'D:\jupyter_project\CNNGRN\DATA\Ecoli\heat_time_3_replice.tsv',
          'D:\jupyter_project\CNNGRN\DATA\Ecoli\oxidativestress_time_3_replice.tsv']
pathY = 'D:\jupyter_project\CNNGRN\DATA\Ecoli\integrated_gold_network.tsv'  # 黄金网络的路径
pathTF = 'D:\jupyter_project\CNNGRN\DATA\Ecoli\TF_from_gold.tsv'  # TF的路径
# 读入原始数据 得到TF序列 和 靶基因序列
TFsID, genesID = CNNGRN_definedFunc.readRawData_Ecoli(pathTF, pathts[0])  # TFsID[0]:(array(['tdca'], dtype=object), 1),  genesID[0]:genesID[0]


# 获取预测的基因对的编号
TFids = []
Targetids = []
for i in range(pred_result.shape[0]):
    TFid = pred_result[i][0]  # TF的编号
    Targetid = pred_result[i][1]  # target gene的编号
    TFids.append(TFid)
    Targetids.append(Targetid)

TFids_name = []
for i in range(len(TFids)):
    a = TFids[i]
    for j in range(len(TFsID)):
        if a==TFsID[j][1]-1:
            b = TFsID[j][0]
            TFids_name.append(b)

Targetids_name = []
for i in range(len(Targetids)):
    s = Targetids[i]
    for j in range(len(genesID)):
        if s == genesID[j][1]-1:
            t = genesID[j][0]
            Targetids_name.append(t)

TFids_name = np.array(TFids_name)
Targetids_name = np.array(Targetids_name)
Targetids_name = Targetids_name.reshape(-1,1)
# 将pred_result与基因对的名称拼接在一起
print(TFids_name.shape)
print(Targetids_name.shape)
name = np.hstack((TFids_name,Targetids_name))
final_pred_result = np.hstack((pred_result,name))
print(final_pred_result.shape)

data1 = pd.DataFrame(final_pred_result)
data1.to_csv('.\\reslutsPredict\\final_pred_result_heat.csv')

print(len(TFids_name))