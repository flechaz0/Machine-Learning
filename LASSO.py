'''
Author: Hang Long
Date: 2025-03-27 20:20:27
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
 
# 生成一些示例数据
np.random.seed(0)
X = 2.5 * np.random.randn(1000) + 1.5   # 生成输入特征X
res = 0.5 * np.random.randn(1000)       # 生成噪声
y = 2 + 0.3 * X + res                   # 实际输出变量y
 
# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
 
# 重塑X_train和X_test为正确的形状
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
 
# 创建Lasso回归模型实例
lasso = Lasso(alpha=0.1)
 
# 拟合模型
lasso.fit(X_train, y_train)
 
# 预测测试集的结果
y_pred = lasso.predict(X_test)
 
# 计算并打印均方误差
mse = mean_squared_error(y_test, y_pred)
print("均方误差(MSE):", mse)
 
# 可视化结果
plt.scatter(X_test, y_test, color='black', label='Actual data')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Lasso model')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Lasso Regression')
plt.legend()
plt.show()
