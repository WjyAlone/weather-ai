# iris_simple.py
"""
鸢尾花分类 - 简化版
适合快速上手和实验
"""

# 导入必要的库
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 创建和训练模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. 预测和评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 5. 输出结果
print(f"测试准确率: {accuracy:.4f}")
print("\n特征重要性:")
for name, importance in zip(iris.feature_names, model.feature_importances_):
    print(f"{name}: {importance:.4f}")

# 6. 简单预测示例
new_sample = [[5.1, 3.5, 1.4, 0.2]]  # 应该是setosa
prediction = model.predict(new_sample)
print(f"\n新样本预测: {iris.target_names[prediction[0]]}")