# **scikit-learn 模块完全指南**

## **目录**
1. [核心设计理念](#核心设计理念)
2. [数据模块](#数据模块)
3. [预处理模块](#预处理模块)
4. [特征工程模块](#特征工程模块)
5. [机器学习模型模块](#机器学习模型模块)
6. [模型选择模块](#模型选择模块)
7. [评估模块](#评估模块)
8. [实用工具模块](#实用工具模块)
9. [应用场景指南](#应用场景指南)

---

## **核心设计理念**

scikit-learn 遵循**一致的API设计**，所有模块都遵循以下核心接口：

```python
# 统一的核心方法
model.fit(X, y)            # 训练
predictions = model.predict(X_new)  # 预测
score = model.score(X_test, y_test)  # 评估

# 统一的核心属性
model.get_params()         # 获取参数
model.set_params(**params)  # 设置参数
```

---

## **数据模块**

### **`sklearn.datasets`** - 数据集加载和生成

#### **内置小数据集（用于教学和测试）**

| 数据集 | 样本数 | 特征数 | 任务类型 | 用途 |
|--------|--------|--------|----------|------|
| `load_iris()` | 150 | 4 | 分类 | 经典分类问题 |
| `load_digits()` | 1797 | 64 | 分类 | 图像分类入门 |
| `load_wine()` | 178 | 13 | 分类 | 多分类问题 |
| `load_breast_cancer()` | 569 | 30 | 分类 | 医学诊断 |

#### **生成数据集（用于实验）**

```python
from sklearn.datasets import make_classification, make_regression, make_blobs

# 分类数据
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,      # 15个相关特征
    n_redundant=5,         # 5个冗余特征
    n_classes=2,
    random_state=42
)

# 回归数据
X_reg, y_reg = make_regression(
    n_samples=500,
    n_features=10,
    noise=0.1,             # 添加噪声
    random_state=42
)

# 聚类数据
X_cluster, y_cluster = make_blobs(
    n_samples=300,
    n_features=2,
    centers=4,             # 4个簇
    cluster_std=0.6,
    random_state=42
)
```

#### **真实世界数据集（需要下载）**

```python
# 回归数据集
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
# 20640个样本，8个特征

# 分类数据集
from sklearn.datasets import fetch_20newsgroups
newsgroups = fetch_20newsgroups(subset='train')
# 文本分类任务
```

---

## **预处理模块**

### **`sklearn.preprocessing`** - 数据预处理

#### **数值特征标准化**

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# 1. 标准正态化（均值=0，方差=1）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 适用：特征服从正态分布

# 2. 最小最大归一化（缩放到[0,1]）
minmax = MinMaxScaler(feature_range=(0, 1))
X_normalized = minmax.fit_transform(X)
# 适用：神经网络、需要固定范围的算法

# 3. 鲁棒标准化（用中位数和分位数）
robust = RobustScaler(quantile_range=(25.0, 75.0))
X_robust = robust.fit_transform(X)
# 适用：数据有异常值的情况
```

#### **分类特征编码**

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# 1. 标签编码（字符串→数字）
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(['cat', 'dog', 'cat', 'bird'])
# 结果：[0, 1, 0, 2]

# 2. 独热编码（创建二进制列）
onehot = OneHotEncoder(sparse=False)
X_onehot = onehot.fit_transform([['cat'], ['dog'], ['bird']])
# 结果：[[1,0,0], [0,1,0], [0,0,1]]

# 3. 序数编码（有序分类）
ordinal = OrdinalEncoder(categories=[['小学', '中学', '大学']])
X_ordinal = ordinal.fit_transform([['大学'], ['小学'], ['中学']])
```

#### **其他预处理方法**

```python
from sklearn.preprocessing import (
    PolynomialFeatures,    # 多项式特征
    FunctionTransformer,   # 自定义转换
    PowerTransformer,      # 幂变换
    QuantileTransformer,   # 分位数变换
    Binarizer             # 二值化
)

# 多项式特征（特征组合）
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_poly = poly.fit_transform(X)
# 创建：x1, x2, x1*x2 等组合特征

# 分位数变换（强制正态分布）
quantile = QuantileTransformer(output_distribution='normal')
X_quantile = quantile.fit_transform(X)
```

---

## **特征工程模块**

### **`sklearn.feature_selection`** - 特征选择

#### **过滤式方法（Filter）**

```python
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif

# 1. 选择K个最佳特征
selector_kbest = SelectKBest(score_func=mutual_info_classif, k=10)
X_selected = selector_kbest.fit_transform(X, y)

# 2. 移除低方差特征
selector_var = VarianceThreshold(threshold=0.01)  # 移除方差<0.01的特征
X_filtered = selector_var.fit_transform(X)

# 3. 基于统计检验
from sklearn.feature_selection import f_classif, chi2
selector_f = SelectKBest(score_func=f_classif, k=15)  # 用于分类
selector_chi2 = SelectKBest(score_func=chi2, k=15)     # 用于分类（非负特征）
```

#### **包裹式方法（Wrapper）**

```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression

# 递归特征消除
estimator = LogisticRegression()
rfe = RFE(estimator, n_features_to_select=5, step=1)
X_rfe = rfe.fit_transform(X, y)
print(rfe.support_)  # 显示哪些特征被选中

# 带交叉验证的RFE
rfecv = RFECV(estimator, cv=5, scoring='accuracy')
rfecv.fit(X, y)
print(f"最优特征数: {rfecv.n_features_}")
```

#### **嵌入式方法（Embedded）**

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

# 基于树模型的特征重要性
model = RandomForestClassifier(n_estimators=100)
selector = SelectFromModel(model, threshold='median', max_features=10)
selector.fit(X, y)
X_embedded = selector.transform(X)
```

### **`sklearn.feature_extraction`** - 特征提取

#### **文本特征提取**

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# 1. 词袋模型（BoW）
vectorizer_bow = CountVectorizer(
    max_features=1000,       # 最大词汇量
    stop_words='english',    # 停用词
    ngram_range=(1, 2)       # 1-gram和2-gram
)
X_bow = vectorizer_bow.fit_transform(text_corpus)

# 2. TF-IDF向量化
vectorizer_tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 3),
    sublinear_tf=True        # 使用1+log(tf)
)
X_tfidf = vectorizer_tfidf.fit_transform(text_corpus)
```

#### **图像特征提取**

```python
from sklearn.feature_extraction import image
import numpy as np

# 从图像中提取小块
img = np.random.rand(100, 100)
patches = image.extract_patches_2d(
    img, 
    patch_size=(8, 8),  # 8x8的小块
    max_patches=100
)
```

### **`sklearn.decomposition`** - 降维

#### **线性降维**

```python
from sklearn.decomposition import PCA, TruncatedSVD

# 1. 主成分分析（PCA）
pca = PCA(n_components=0.95)  # 保留95%方差
X_pca = pca.fit_transform(X)
print(f"解释方差比例: {pca.explained_variance_ratio_}")

# 2. 截断SVD（用于稀疏矩阵）
svd = TruncatedSVD(n_components=50, algorithm='randomized')
X_svd = svd.fit_transform(X_sparse)
```

#### **非线性降维**

```python
from sklearn.decomposition import KernelPCA, NMF, FactorAnalysis

# 1. 核PCA（处理非线性数据）
kpca = KernelPCA(n_components=10, kernel='rbf', gamma=0.1)
X_kpca = kpca.fit_transform(X)

# 2. 非负矩阵分解（NMF）
nmf = NMF(n_components=10, init='random', random_state=42)
X_nmf = nmf.fit_transform(X_nonnegative)

# 3. 因子分析（FA）
fa = FactorAnalysis(n_components=8, random_state=42)
X_fa = fa.fit_transform(X)
```

---

## **机器学习模型模块**

### **`sklearn.linear_model`** - 线性模型

| 模型 | 回归/分类 | 正则化 | 适用场景 |
|------|-----------|--------|----------|
| `LinearRegression` | 回归 | 无 | 线性关系强，无多重共线性 |
| `Ridge` | 回归 | L2 | 特征多重共线性 |
| `Lasso` | 回归 | L1 | 特征选择，稀疏解 |
| `ElasticNet` | 回归 | L1+L2 | 结合L1和L2优点 |
| `LogisticRegression` | 分类 | L1/L2 | 二分类/多分类 |
| `SGDClassifier` | 分类 | L1/L2 | 大数据集，在线学习 |

```python
# 线性回归对比
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.01),
    'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5)
}

# 逻辑回归示例
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(
    penalty='l2',          # 正则化类型
    C=1.0,                 # 正则化强度倒数
    solver='lbfgs',        # 优化算法
    max_iter=1000,
    multi_class='ovr'      # 多分类策略
)
```

### **`sklearn.svm`** - 支持向量机

#### **核心参数解析**
```python
from sklearn.svm import SVC, SVR, LinearSVC

# 非线性SVM（使用核技巧）
svm_rbf = SVC(
    kernel='rbf',          # 径向基函数核
    C=1.0,                 # 正则化参数
    gamma='scale',         # 核函数系数
    probability=True,      # 启用概率估计
    decision_function_shape='ovr'  # 多分类策略
)

# 线性SVM（更高效）
svm_linear = LinearSVC(
    penalty='l2',
    C=1.0,
    loss='squared_hinge',
    dual=True  # 对偶问题（样本数<特征数时设为False）
)

# 支持向量回归
svr = SVR(
    kernel='rbf',
    C=1.0,
    epsilon=0.1  # epsilon-insensitive损失函数参数
)
```

#### **核函数选择指南**
| 核函数 | 公式 | 适用场景 | 复杂度 |
|--------|------|----------|--------|
| `linear` | K(x,y) = x·y | 线性可分，特征多 | O(n_features) |
| `poly` | K(x,y) = (γx·y + r)^d | 有一定非线性 | O(n_features^d) |
| `rbf` | K(x,y) = exp(-γ||x-y||²) | 复杂非线性 | O(n_samples²) |
| `sigmoid` | K(x,y) = tanh(γx·y + r) | 神经网络相似 | O(n_samples²) |

### **`sklearn.ensemble`** - 集成学习

#### **Bagging方法（并行）**
```python
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

# 1. Bagging分类器
bagging = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,      # 每次采样的比例
    max_features=0.8,     # 每次采样的特征比例
    bootstrap=True,       # 有放回采样
    bootstrap_features=False,
    n_jobs=-1             # 使用所有CPU核心
)

# 2. 随机森林（特殊Bagging）
rf = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',     # 或 'entropy'
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',  # 每棵树使用sqrt(n_features)个特征
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)
```

#### **Boosting方法（串行）**
```python
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

# 1. AdaBoost（自适应增强）
adaboost = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),  # 弱分类器
    n_estimators=100,
    learning_rate=1.0,    # 学习率
    algorithm='SAMME.R'   # 多分类算法
)

# 2. 梯度提升
gbdt = GradientBoostingClassifier(
    loss='deviance',      # 或 'exponential'
    learning_rate=0.1,
    n_estimators=100,
    subsample=1.0,        # 样本采样比例
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,
    random_state=42
)

# 3. 直方图梯度提升（大数据集）
from sklearn.ensemble import HistGradientBoostingClassifier
hgb = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_depth=8,
    categorical_features=None,  # 指定分类特征索引
    early_stopping=True        # 早停
)
```

#### **投票和堆叠**
```python
from sklearn.ensemble import VotingClassifier, StackingClassifier

# 1. 投票分类器（硬投票/软投票）
voting = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier()),
        ('svc', SVC(probability=True)),
        ('gbdt', GradientBoostingClassifier())
    ],
    voting='soft',  # 'hard'或'soft'
    weights=[2, 1, 1]  # 模型权重
)

# 2. 堆叠分类器
stacking = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier()),
        ('svc', SVC(probability=True))
    ],
    final_estimator=LogisticRegression(),
    cv=5,            # 交叉验证折数
    n_jobs=-1
)
```

### **`sklearn.tree`** - 决策树

```python
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

# 决策树分类器
dt = DecisionTreeClassifier(
    criterion='gini',      # 或 'entropy'
    splitter='best',       # 或 'random'
    max_depth=None,        # 树的最大深度
    min_samples_split=2,   # 内部节点分裂所需最小样本数
    min_samples_leaf=1,    # 叶节点所需最小样本数
    max_features=None,     # 寻找最佳分裂时考虑的特征数
    random_state=42,
    ccp_alpha=0.0          # 最小代价复杂度剪枝参数
)

# 可视化决策树
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plot_tree(
    dt,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.show()

# 导出决策规则
tree_rules = export_text(
    dt,
    feature_names=feature_names
)
print(tree_rules)
```

### **`sklearn.naive_bayes`** - 朴素贝叶斯

| 模型 | 数据分布假设 | 适用场景 | 特征要求 |
|------|-------------|----------|----------|
| `GaussianNB` | 高斯分布 | 连续特征 | 特征连续，近似正态分布 |
| `MultinomialNB` | 多项分布 | 计数数据 | 特征非负，如文本词频 |
| `BernoulliNB` | 伯努利分布 | 二值特征 | 特征为0/1 |
| `ComplementNB` | 补集分布 | 不平衡文本 | 改进MultinomialNB |
| `CategoricalNB` | 分类分布 | 分类特征 | 离散特征 |

```python
# 文本分类示例
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

text_clf = make_pipeline(
    TfidfVectorizer(stop_words='english'),
    MultinomialNB(alpha=1.0)  # 平滑参数
)
text_clf.fit(train_texts, train_labels)
```

### **`sklearn.neighbors`** - 最近邻方法

```python
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

# K近邻分类
knn = KNeighborsClassifier(
    n_neighbors=5,           # 邻居数量
    weights='uniform',       # 'uniform'或'distance'
    algorithm='auto',        # 'auto', 'ball_tree', 'kd_tree', 'brute'
    leaf_size=30,           # 树结构参数
    p=2,                    # 距离度量（1:曼哈顿，2:欧式）
    metric='minkowski',     # 距离度量
    n_jobs=-1
)

# 最近邻搜索（聚类/推荐系统）
nn = NearestNeighbors(
    n_neighbors=10,
    radius=1.0,
    algorithm='ball_tree',
    metric='cosine'         # 余弦相似度
)
nn.fit(X)
distances, indices = nn.kneighbors(query_point)
```

### **`sklearn.cluster`** - 聚类算法

#### **基于距离的聚类**
```python
from sklearn.cluster import KMeans, AgglomerativeClustering

# 1. K-means聚类
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',      # 初始化方法
    n_init=10,             # 不同初始化的运行次数
    max_iter=300,
    tol=1e-4,
    random_state=42,
    algorithm='lloyd'      # 'lloyd'或'elkan'
)

# 2. 层次聚类
hierarchical = AgglomerativeClustering(
    n_clusters=None,       # 可以指定距离阈值代替
    affinity='euclidean',  # 距离度量
    linkage='ward',        # 'ward', 'complete', 'average', 'single'
    distance_threshold=0.5 # 距离阈值
)
```

#### **基于密度的聚类**
```python
from sklearn.cluster import DBSCAN, OPTICS

# DBSCAN（发现任意形状的簇）
dbscan = DBSCAN(
    eps=0.5,               # 邻域半径
    min_samples=5,         # 核心点的最小邻居数
    metric='euclidean',
    algorithm='auto',      # 'auto', 'ball_tree', 'kd_tree', 'brute'
    leaf_size=30,
    n_jobs=-1
)

# OPTICS（改进的DBSCAN）
optics = OPTICS(
    min_samples=5,
    max_eps=float('inf'),
    metric='minkowski',
    cluster_method='xi',   # 'xi'或'dbscan'
    xi=0.05,
    min_cluster_size=None
)
```

#### **其他聚类方法**
```python
from sklearn.cluster import SpectralClustering, AffinityPropagation

# 谱聚类（基于图论）
spectral = SpectralClustering(
    n_clusters=5,
    eigen_solver=None,
    affinity='rbf',        # 'nearest_neighbors', 'rbf', 'precomputed'
    gamma=1.0,
    n_neighbors=10,
    assign_labels='kmeans'  # 'kmeans'或'discretize'
)

# 亲和传播（基于消息传递）
affinity = AffinityPropagation(
    damping=0.5,           # 阻尼系数(0.5-1.0)
    max_iter=200,
    convergence_iter=15,
    preference=None,       # 偏向参数
    affinity='euclidean'
)
```

### **`sklearn.neural_network`** - 神经网络

```python
from sklearn.neural_network import MLPClassifier, MLPRegressor

# 多层感知机（MLP）
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50, 25),  # 隐藏层结构
    activation='relu',                 # 激活函数
    solver='adam',                     # 优化器
    alpha=0.0001,                      # L2正则化参数
    batch_size='auto',
    learning_rate='adaptive',          # 学习率策略
    learning_rate_init=0.001,
    max_iter=300,
    shuffle=True,
    random_state=42,
    tol=1e-4,
    early_stopping=True,               # 早停
    validation_fraction=0.1,           # 验证集比例
    beta_1=0.9,                        # Adam参数
    beta_2=0.999,
    epsilon=1e-8,
    n_iter_no_change=10
)
```

---

## **模型选择模块**

### **`sklearn.model_selection`** - 模型选择和验证

#### **数据集划分**
```python
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

# 1. 简单划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,          # 测试集比例
    random_state=42,        # 随机种子
    stratify=y,             # 分层采样（保持类别比例）
    shuffle=True            # 打乱数据
)

# 2. 分层K折交叉验证（保持类别比例）
skf = StratifiedKFold(
    n_splits=5,            # K值
    shuffle=True,
    random_state=42
)

# 3. 时间序列交叉验证
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

#### **超参数调优**
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# 1. 网格搜索（穷举）
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(),
    param_grid=param_grid,
    scoring='accuracy',     # 评估指标
    cv=5,                  # 交叉验证折数
    verbose=1,             # 输出详细程度
    n_jobs=-1,             # 并行数
    refit=True             # 用最佳参数重新训练
)
grid_search.fit(X_train, y_train)
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳分数: {grid_search.best_score_:.3f}")

# 2. 随机搜索（效率更高）
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 15),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4)  # 0.6到1.0
}

random_search = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(),
    param_distributions=param_dist,
    n_iter=50,             # 随机采样次数
    scoring='accuracy',
    cv=5,
    random_state=42,
    n_jobs=-1
)
```

#### **学习曲线和验证曲线**
```python
from sklearn.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt
import numpy as np

# 1. 学习曲线（检查欠拟合/过拟合）
train_sizes, train_scores, val_scores = learning_curve(
    estimator=model,
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# 2. 验证曲线（选择超参数）
param_range = [0.001, 0.01, 0.1, 1.0, 10.0]
train_scores, val_scores = validation_curve(
    estimator=model,
    X=X_train,
    y=y_train,
    param_name='C',        # 要调整的参数
    param_range=param_range,
    cv=5,
    scoring='accuracy'
)

# 可视化
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_sizes, train_scores.mean(axis=1), label='训练')
plt.plot(train_sizes, val_scores.mean(axis=1), label='验证')
plt.xlabel('训练样本数')
plt.ylabel('准确率')
plt.legend()
plt.grid(True)
```

---

## **评估模块**

### **`sklearn.metrics`** - 模型评估

#### **分类指标**
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)

# 1. 基础指标
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')  # 宏平均
recall = recall_score(y_true, y_pred, average='weighted')     # 加权平均
f1 = f1_score(y_true, y_pred, average='micro')               # 微平均

# 2. 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
# [[TP, FN],
#  [FP, TN]]

# 3. 分类报告（综合指标）
report = classification_report(
    y_true, y_pred,
    target_names=class_names,
    digits=3
)
print(report)

# 4. ROC和PR曲线
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
roc_auc = roc_auc_score(y_true, y_pred_proba)

precision_vals, recall_vals, thresholds_pr = precision_recall_curve(
    y_true, y_pred_proba
)
```

#### **回归指标**
```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, mean_squared_log_error
)

mse = mean_squared_error(y_true, y_pred, squared=True)
rmse = mean_squared_error(y_true, y_pred, squared=False)  # 平方根
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
evs = explained_variance_score(y_true, y_pred)

# 可视化回归结果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title(f'R² = {r2:.3f}, RMSE = {rmse:.3f}')
plt.grid(True)
plt.show()
```

#### **聚类指标**
```python
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score
)

# 无监督指标（不需要真实标签）
silhouette = silhouette_score(X, labels, metric='euclidean')
db_index = davies_bouldin_score(X, labels)
ch_index = calinski_harabasz_score(X, labels)

# 有监督指标（需要真实标签）
ari = adjusted_rand_score(y_true, labels)      # 调整兰德指数
nmi = normalized_mutual_info_score(y_true, labels)  # 标准化互信息
```

#### **自定义评估函数**
```python
from sklearn.metrics import make_scorer
import numpy as np

# 创建自定义评分函数
def custom_metric(y_true, y_pred):
    """自定义业务指标"""
    # 示例：加权准确率
    weights = np.array([1.0, 2.0, 1.5])  # 不同类别权重
    class_correct = np.bincount(y_true[y_true == y_pred])
    weighted_acc = np.sum(class_correct * weights) / len(y_true)
    return weighted_acc

custom_scorer = make_scorer(custom_metric, greater_is_better=True)

# 在网格搜索中使用
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring=custom_scorer,  # 使用自定义评分
    cv=5
)
```

---

## **实用工具模块**

### **`sklearn.pipeline`** - 机器学习流水线

```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 1. 创建列转换器（不同列不同处理）
numeric_features = ['age', 'income', 'credit_score']
categorical_features = ['education', 'marital_status']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 2. 创建完整流水线
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(k=10)),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

# 3. 训练和预测（像单个模型一样使用）
full_pipeline.fit(X_train, y_train)
y_pred = full_pipeline.predict(X_test)

# 4. 访问流水线中的步骤
preprocessor = full_pipeline.named_steps['preprocessor']
classifier = full_pipeline.named_steps['classifier']
```

### **`sklearn.externals.joblib`** - 模型持久化

```python
import joblib

# 保存模型
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# 加载模型
loaded_model = joblib.load('model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# 保存和加载整个流水线
joblib.dump(full_pipeline, 'pipeline.pkl')
loaded_pipeline = joblib.load('pipeline.pkl')

# 在生产环境中使用
def predict_new_data(input_data):
    """API服务中的预测函数"""
    # 加载模型
    model = joblib.load('model.pkl')
    
    # 预处理输入
    input_df = pd.DataFrame([input_data])
    input_processed = loaded_scaler.transform(input_df)
    
    # 预测
    prediction = model.predict(input_processed)
    probability = model.predict_proba(input_processed)
    
    return {
        'prediction': int(prediction[0]),
        'probability': probability[0].tolist()
    }
```

### **`sklearn.utils`** - 实用工具函数

```python
from sklearn.utils import (
    shuffle,                # 打乱数据
    resample,               # 重采样
    class_weight,           # 计算类别权重
    check_array,            # 检查数组
    check_X_y,              # 检查X和y
    compute_class_weight,   # 计算类别权重
    multiclass,             # 多分类工具
    safe_indexing           # 安全索引
)

# 1. 处理不平衡数据
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

# 2. 打乱数据
X_shuffled, y_shuffled = shuffle(X, y, random_state=42)

# 3. 重采样（Bootstrap）
X_resampled, y_resampled = resample(
    X, y,
    n_samples=1000,
    random_state=42,
    stratify=y
)

# 4. 检查数据
X_checked = check_array(X, dtype='numeric', accept_sparse=True)
X_checked, y_checked = check_X_y(X, y, multi_output=True)
```

---

## **应用场景指南**

### **场景1：分类问题流程**

```python
# 1. 导入所有需要的模块
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 2. 加载和准备数据
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. 创建流水线
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 4. 定义参数网格
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 5, 10],
    'classifier__min_samples_split': [2, 5, 10]
}

# 5. 网格搜索
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

# 6. 评估最佳模型
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
```

### **场景2：回归问题流程**

```python
# 1. 回归问题专用模块
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 2. 加载数据
housing = fetch_california_housing()
X, y = housing.data, housing.target

# 3. 创建回归流水线
pipeline = Pipeline([
    ('scaler', RobustScaler()),  # 鲁棒标准化（对异常值不敏感）
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# 4. 交叉验证评估
cv_scores = cross_val_score(
    pipeline,
    X, y,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
rmse_scores = np.sqrt(-cv_scores)
print(f"RMSE: {rmse_scores.mean():.3f} (±{rmse_scores.std():.3f})")

# 5. 最终训练和评估
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(f"测试集 R²: {r2_score(y_test, y_pred):.3f}")
print(f"测试集 RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
```

### **场景3：聚类分析流程**

```python
# 1. 聚类分析专用模块
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 2. 生成数据
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# 3. 寻找最佳K值（肘部法则）
inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# 4. 可视化肘部曲线
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.title('Elbow Method')

# 5. 轮廓系数评估
silhouette_scores = []
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, 'ro-')
plt.xlabel('K')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.tight_layout()
plt.show()

# 6. 选择最佳K并进行最终聚类
best_k = silhouette_scores.index(max(silhouette_scores)) + 2
final_kmeans = KMeans(n_clusters=best_k, random_state=42)
final_labels = final_kmeans.fit_predict(X_scaled)
```

### **场景4：文本分类流程**

```python
# 1. 文本处理专用模块
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 2. 加载文本数据
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

# 3. 创建文本分类流水线
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),  # 使用unigram和bigram
        min_df=5,            # 最小文档频率
        max_df=0.7           # 最大文档频率
    )),
    ('clf', MultinomialNB(alpha=0.1))
])

# 4. 训练和评估
text_clf.fit(newsgroups_train.data, newsgroups_train.target)
y_pred = text_clf.predict(newsgroups_test.data)

print(classification_report(
    newsgroups_test.target,
    y_pred,
    target_names=newsgroups_test.target_names
))

# 5. 查看最重要的特征
feature_names = text_clf.named_steps['tfidf'].get_feature_names_out()
coefs = text_clf.named_steps['clf'].coef_
for i, category in enumerate(newsgroups_test.target_names):
    top10 = np.argsort(coefs[i])[-10:]  # 权重最高的10个词
    print(f"\n{category}:")
    print(", ".join(feature_names[top10]))
```

---

## **学习路径建议**

### **第1阶段：基础掌握（1-2周）**
1. **`datasets`** - 熟悉内置数据集
2. **`model_selection`** - 掌握数据划分和交叉验证
3. **`preprocessing`** - 学会数据标准化和编码
4. **`metrics`** - 理解各种评估指标

### **第2阶段：模型学习（2-3周）**
1. **`linear_model`** - 逻辑回归、线性回归
2. **`ensemble`** - 随机森林、梯度提升
3. **`svm`** - 支持向量机
4. **`neighbors`** - K近邻

### **第3阶段：进阶应用（3-4周）**
1. **`pipeline`** - 构建完整工作流
2. **`feature_selection/extraction`** - 特征工程
3. **`decomposition`** - 降维技术
4. **`cluster`** - 聚类分析

### **第4阶段：实战项目（持续）**
1. **表格数据** - 使用完整流水线解决分类/回归问题
2. **文本数据** - 结合TfidfVectorizer和分类器
3. **图像数据** - 使用特征提取结合传统模型
4. **时间序列** - 使用时间特征工程

---

## **最佳实践总结**

1. **始终使用流水线**：避免数据泄露，确保代码可复用
2. **交叉验证调参**：使用GridSearchCV或RandomizedSearchCV
3. **特征工程是关键**：花时间在数据处理上
4. **模型可解释性**：使用决策树、逻辑回归等可解释模型作为基线
5. **从简单开始**：先尝试线性模型，再考虑复杂模型
6. **监控过拟合**：观察训练和验证性能差异
7. **保存和加载模型**：使用joblib进行模型持久化

这个完整的指南涵盖了scikit-learn的所有核心模块，按照这个框架学习，可以系统地掌握scikit-learn的强大功能！