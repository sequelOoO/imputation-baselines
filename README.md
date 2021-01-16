Transportation Data Imputation Baselines

## 1、研究背景
在大数据时代背景下，随着越来越多的传感器投入到交通状况的检测中，大量的相关数据被收集。这些数据经过分析处理可以帮助指导人类的生产和生活，比如在车辆出行规划中避开拥堵路线，在市政建设中科学高效地做扩宽道路或新建道路的决策以解决道路拥堵问题等。
### 1.1.产生原因
数据出现丢失的情况在收集数据的过程中不可避免，有人为因素、自然因素等，比如停电、传感器故障、网络传输出错、存储错误等。
### 1.2.研究意义
不完整的数据对后续数据的分析处理会造成较大影响，比如一些预测模型的表现，会因为数据的缺失而变差。当数据缺失率达到一定程度时，模型甚至会不能正常工作。因此需要用一些方法对缺失的数据进行补全，并尽可能使补全的值接近真实值，用补全缺失值的方式降低数据因缺失带来的影响，增加数据的价值。
## 2、现有方法
缺失数据补全仍然是一个比较热门的研究方向，各种各样的方法被提出来试图还原缺失值。现有的方法中，一般可分为以下三类，一是统计学的方法，包括平均值、中值、众数等，二是传统机器学习方法，包括PPCA（概率主成分分析）、k-NN（k最近邻）、BNs（贝叶斯网络）等，还有一系列张量分解（矩阵分解）的方法，如BGCP（贝叶斯高斯张量分解）、BATF（贝叶斯增广张量分解）、BPMF（贝叶斯概率矩阵分解）、BTMF（贝叶斯时间矩阵分解）、BTRMF（贝叶斯时间正则化矩阵分解）、CP_ALS等。三是基于神经网络的方法，像GINN（graph imputation neural network，Indro Spinelli et al.，2019，Missing Data Imputation with Adversarially-trained Graph Convolutional Networks）、E2EGAN（Yonghong Luo et al.，2019，E2GAN: End-to-End Generative Adversarial Network for Multivariate Time Series Imputation）等。
### 2.1. 统计学方法
平均值、中值方法分别将时间序列中一定的窗口内除缺失值外的所有值求平均与求中值后代替缺失的值。
众数方法是用序列中出现次数最多的值代替缺失值。
### 2.2 传统机器学习方法
### 2.2.1 PPCA方法
该方法假设假设所有的样本点都取样于某个分布 x ∈ ，对于每个点都有一个 与之对应，取样于某个分布 z ∈ ，满足以下条件：

### 2.2.2 KNN方法
knn方法通过两个步骤来补全丢失数据，1、选择。从数据集中选择k个与目标向量最接近的向量，测量模式相似性的方法有欧几里得距离Euclidean distance (Ed)和皮尔森相关性Pearson correlation (Pc)。2、补全。补全缺失值通过以下公式进行，有点类似于k个向量中对应值的加权平均。

### 2.2.3 BNs方法
该方法在已知t时刻前m个值的情况下计算t时刻的期望。

### 2.2.4BGCP
是一种以贝叶斯推断 (Bayesian inference) 、高斯假设为基础的张量分解技术。
高斯假设下的张量模型：
![photo](https://github.com/sequelOoO/imputation-baselines/blob/main/img/%E5%9B%BE%E7%89%871.png)

模型参数的先验分布：
![photo](https://github.com/sequelOoO/imputation-baselines/blob/main/img/%E5%9B%BE%E7%89%872.png)

超参数的先验分布：
![photo](https://github.com/sequelOoO/imputation-baselines/blob/main/img/%E5%9B%BE%E7%89%873.png)

其中，上面出现的向量如在没有特殊说明的情况下，都表示列向量； α 和 β 也是模型的超参数。
对于模型参数，例如  ，它的似然来自于观测值  ，先验是一个多元正态分布，因此，  的后验分布仍然是一个多元正态分布，即  ，后验的参数如下：

其中，向量  ，符号  表示点乘 (element-wise product)，即对于任意  ，满足  ；若 Ω 表示张量  中所有被观测到的元素的索引集合，则后验参数中的求和符号下标  是切片 (slice)  中被观测到的元素的索引。
事实上，模型参数  和  的后验分布参数与  类似，即
 ，后验分布的参数为

，后验分布的参数为

其中，向量

待补充...
2.2.5BATF
2.2.6BPMF
2.2.7BTMF
2.2.8BTRMF
2.2.9CP_ALS
2.3基于神经网络的方法
2.3.1GINN
2.3.2E2EGAN
## 3、实验
此次实验用到的是西雅图高速公路车速数据集（Seattle-data-set），数据集收集了西雅图高速公路上323个侦测点，连续28天的检测值，其中每天检测的时间间隔为5分钟，单个检测点每天产生288（24*60/5=288）个数据。
实现的实验方法为统计学习方法和传统机器学习方法，包括平均数、中位数、众数、PPCA、k-NN、BNs、BGCP、BATF、BPMF、BTMF、BTRMF、CP_ALS等12种方法。
实验结果用mape和rmse两种评价指标来评价。
mape：平均绝对百分比误差（Mean Absolute Percentage Error），mape的值越小，代表模型表现效果越好，mape的值越大，代表模型表现效果越差。从公式中可以看出，当真实值存在为0的情况时该公式不能用。

rmse：均方根误差（Root Mean Square Error），rmse值越小，模型效果越好，反之则越差。

从最后的折线图可以看出，BTRMF方法表现最好，众数方法表现最差。再进一步观察发现，传统机器学习方法的表现，普遍优于简单的统计学的方法。

补全方法使用示例：

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

# 两种评价指标，mape和rmse，代码实现：

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
knn_mape8 = mape(dense_mat[pos8],knn_res8[pos8])

# 结果展示。
print("knn mape ,missing rate\n20% {}\n40% {}\n60% {}\n80% {}".format(knn_mape2,knn_mape4,knn_mape6,knn_mape8))
# knn mape ,missing rate
# 20% 8.552323762998148
# 40% 9.362702031879724
# 60% 10.360443113147243
# 80% 12.368586561731675


```

