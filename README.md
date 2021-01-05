# imputation-baselines
一些补全方面的baselines的整理

knn方法
knn方法通过两个步骤来补全丢失数据，1、选择。从数据集中选择k个与目标向量最接近的向量，测量模式相似性的方法有欧几里得距离Euclidean distance (Ed)和皮尔森相关性Pearson correlation (Pc)。2、补全。补全缺失值通过以下公式进行，有点类似于k个向量中对应值的加权平均。

<div align=center>![img](https://github.com/sequelOoO/imputation-baselines/blob/main/img/%E5%85%AC%E5%BC%8F1.png)

BNs方法
该方法在已知t时刻前m个值的情况下计算t时刻的期望。

<div align=center>![img](https://github.com/sequelOoO/imputation-baselines/blob/main/img/%E5%85%AC%E5%BC%8F%E4%BA%8C.png)

PPCA方法
该方法假设假设所有的样本点都取样于某个分布 x ∈ ，对于每个点都有一个 与之对应，取样于某个分布 z ∈ ，满足以下条件

<div align=center>![img](https://github.com/sequelOoO/imputation-baselines/blob/main/img/%E5%85%AC%E5%BC%8F%E4%B8%89.png)

使用示例：
```
# 导入imputer，然后实例化对象
import numpy as np
import pandas as pd
import imputer
imp = imputer.Imputer()

# 导入数据，dense_mat是完整的原数据（一个矩阵），rm是和原数据同型的矩阵，值是在0-1之间均匀分布的随机数。将随机数矩阵保存下来而不是每次运行时再随机生成，是为了保证在不同机器上运行的结果都是相同的。
dense_mat = pd.read_csv('./datasets/Seattle-data-set/mat.csv',index_col=0)
rm = pd.read_csv('./datasets/Seattle-data-set/RM_mat.csv',index_col=0)
dense_mat = dense_mat.values
rm = rm.values

# 从随机矩阵中获得不同随机缺失率的0,1矩阵，矩阵的值为0或1.
binary_mat2 = np.round(rm + 0.5 - 0.2)
binary_mat4 = np.round(rm+0.5 -0.4)
binary_mat6= np.round(rm+0.5-0.6)
binary_mat8= np.round(rm+0.5-0.8)

# 将01矩阵复制一遍，再将矩阵中为0的值用np.nan代替。
nan_mat2 = binary_mat2.copy()
nan_mat4 = binary_mat4.copy()
nan_mat6 = binary_mat6.copy()
nan_mat8 = binary_mat8.copy()

nan_mat2[nan_mat2 == 0] = np.nan
nan_mat4[nan_mat4 == 0] = np.nan
nan_mat6[nan_mat6 == 0] = np.nan
nan_mat8[nan_mat8 == 0] = np.nan

# 矩阵相乘，得到稀疏矩阵，缺失值用np.nan代替。
sparse_mat2 = np.multiply(nan_mat2, dense_mat)
sparse_mat4 = np.multiply(nan_mat4, dense_mat)
sparse_mat6 = np.multiply(nan_mat6, dense_mat)
sparse_mat8 = np.multiply(nan_mat8, dense_mat)

# 调用imp的knn方法，用knn方法来补全数据，得到补全后的结果。
knn_res2 = imp.knn(sparse_mat2)
knn_res4 = imp.knn(sparse_mat4)
knn_res6 = imp.knn(sparse_mat6)
knn_res8 = imp.knn(sparse_mat8)

# 两种评价指标，mape和rmse
from sklearn import metrics
def mape(y_true,y_pred):
    return np.mean(np.abs(y_pred-y_true)/y_true)*100
def rmse(y_true,y_pred):
return np.sqrt(metrics.mean_squared_error(y_pred,y_true))


# 找到矩阵中可以用于评价的值的位置，必须满足：在稀疏矩阵中是缺失的，在原矩阵中的值不为0。
pos2 = np.where((dense_mat != 0) & (binary_mat2 == 0))
pos4 = np.where((dense_mat != 0) & (binary_mat4 == 0))
pos6 = np.where((dense_mat != 0) & (binary_mat6 == 0))
pos8 = np.where((dense_mat != 0) & (binary_mat8 == 0))

# 计算knn方法对不同缺失率的补全结果的mape。
knn_mape2 = mape(dense_mat[pos2],knn_res2[pos2])
knn_mape4 = mape(dense_mat[pos4],knn_res4[pos4])
knn_mape6 = mape(dense_mat[pos6],knn_res6[pos6])
knn_mape8 = mape(dense_mat[pos8],knn_res8[pos8])```

# 结果展示。
print("knn mape ,missing rate\n20% {}\n40% {}\n60% {}\n80% {}".format(knn_mape2,knn_mape4,knn_mape6,knn_mape8))
# knn mape ,missing rate
# 20% 8.552323762998148
# 40% 9.362702031879724
# 60% 10.360443113147243
# 80% 12.368586561731675
```
