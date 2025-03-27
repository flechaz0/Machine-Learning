'''
Author: Hang Long
Date: 2025-02-25 20:59:25
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
 
# 加载数据
iris = load_iris()
X = iris.data
print('type of X: ',type(X))
y = iris.target
print('type of y: ',type(y))
# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# 创建随机森林模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)
 
# 训练模型
clf.fit(X_train, y_train)
 
# 预测测试集
y_pred = clf.predict(X_test)
 
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
 
# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
 
# 显示特征重要性
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
 
plt.figure()
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], color="r", align="center")
plt.xticks(range(X_train.shape[1]), np.array(iris.feature_names)[indices], rotation=90)  # 修改这里，确保索引和标签匹配
plt.xlim([-1, X_train.shape[1]])
plt.show()
 
# 预测新数据（假设的样本）
new_samples = [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]
new_predictions = clf.predict(new_samples)
print("New Predictions:", new_predictions)