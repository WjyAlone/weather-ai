# **PyTorch 模块完全指南**

## **目录**
- [**PyTorch 模块完全指南**](#pytorch-模块完全指南)
  - [**目录**](#目录)
  - [**层（Layers）**](#层layers)
    - [**1. 线性/全连接层**](#1-线性全连接层)
    - [**2. 卷积层**](#2-卷积层)
    - [**3. 池化层**](#3-池化层)
    - [**4. 填充层**](#4-填充层)
  - [**容器（Containers）**](#容器containers)
    - [**1. 顺序容器**](#1-顺序容器)
    - [**2. 示例对比**](#2-示例对比)
  - [**激活函数（Activation Functions）**](#激活函数activation-functions)
  - [**损失函数（Loss Functions）**](#损失函数loss-functions)
    - [**1. 回归损失**](#1-回归损失)
    - [**2. 分类损失**](#2-分类损失)
    - [**3. 选择指南**](#3-选择指南)
  - [**优化器（Optimizers）**](#优化器optimizers)
    - [**1. 基础优化器**](#1-基础优化器)
    - [**2. 参数对比**](#2-参数对比)
    - [**3. 选择策略**](#3-选择策略)
  - [**学习率调度器（Learning Rate Schedulers）**](#学习率调度器learning-rate-schedulers)
    - [**1. 基于步数的调度器**](#1-基于步数的调度器)
    - [**2. 基于指标的调度器**](#2-基于指标的调度器)
    - [**3. 使用示例**](#3-使用示例)
  - [**归一化层（Normalization Layers）**](#归一化层normalization-layers)
  - [**循环神经网络层（RNN Layers）**](#循环神经网络层rnn-layers)
    - [**1. 基础RNN层**](#1-基础rnn层)
    - [**2. 参数详解**](#2-参数详解)
    - [**3. 选择指南**](#3-选择指南-1)
  - [**Transformer层（Transformer Layers）**](#transformer层transformer-layers)
    - [**1. 核心组件**](#1-核心组件)
    - [**2. 使用示例**](#2-使用示例)
    - [**3. 适用场景**](#3-适用场景)
  - [**实用工具（Utility Modules）**](#实用工具utility-modules)
    - [**1. Dropout层**](#1-dropout层)
    - [**2. 嵌入层**](#2-嵌入层)
    - [**3. 距离函数**](#3-距离函数)
    - [**4. 其他工具**](#4-其他工具)
  - [**功能接口（Functional Interface）**](#功能接口functional-interface)
    - [**1. 什么是Functional？**](#1-什么是functional)
    - [**2. 常用Functional函数**](#2-常用functional函数)
    - [**3. 使用场景**](#3-使用场景)
  - [**模块选择快速参考表**](#模块选择快速参考表)
  - [**总结**](#总结)

---

## **层（Layers）**

### **1. 线性/全连接层**
| 模块 | 作用 | 主要参数 | 输出形状 |
|------|------|---------|---------|
| `nn.Linear` | 全连接线性变换 | `in_features`, `out_features`, `bias` | `(N, *, in_features) → (N, *, out_features)` |
| `nn.Bilinear` | 双线性变换 | `in1_features`, `in2_features`, `out_features`, `bias` | `(N, *, in1)`, `(N, *, in2) → (N, *, out)` |
| `nn.Identity` | 恒等映射（占位层） | 无 | 输入=输出 |

**区别**：
- `Linear`：单输入线性变换 `y = xA^T + b`
- `Bilinear`：双输入线性组合 `y = x1A x2^T + b`
- `Identity`：不做任何操作，常用于网络架构设计

### **2. 卷积层**
| 模块 | 作用 | 主要参数 | 适用数据 |
|------|------|---------|---------|
| `nn.Conv1d` | 一维卷积 | `in_channels`, `out_channels`, `kernel_size` | 音频、时序数据 |
| `nn.Conv2d` | 二维卷积 | `in_channels`, `out_channels`, `kernel_size` | 图像数据 |
| `nn.Conv3d` | 三维卷积 | `in_channels`, `out_channels`, `kernel_size` | 视频、体积数据 |
| `nn.ConvTranspose1d` | 一维转置卷积 | 同Conv1d | 上采样、生成模型 |
| `nn.ConvTranspose2d` | 二维转置卷积 | 同Conv2d | 图像上采样、生成模型 |
| `nn.ConvTranspose3d` | 三维转置卷积 | 同Conv3d | 体积数据上采样 |

**共同参数**：`stride`, `padding`, `dilation`, `groups`, `bias`

### **3. 池化层**
| 模块 | 作用 | 主要参数 | 特点 |
|------|------|---------|------|
| `nn.MaxPool1d/2d/3d` | 最大池化 | `kernel_size`, `stride`, `padding` | 取窗口内最大值 |
| `nn.AvgPool1d/2d/3d` | 平均池化 | `kernel_size`, `stride`, `padding` | 取窗口内平均值 |
| `nn.AdaptiveMaxPool1d/2d/3d` | 自适应最大池化 | `output_size` | 输出指定大小 |
| `nn.AdaptiveAvgPool1d/2d/3d` | 自适应平均池化 | `output_size` | 输出指定大小 |
| `nn.MaxUnpool1d/2d/3d` | 最大反池化 | `kernel_size`, `stride`, `padding` | MaxPool的逆操作 |
| `nn.FractionalMaxPool2d` | 分数步长最大池化 | `kernel_size`, `output_size` | 输出为分数 |
| `nn.LPPool1d/2d` | Lp范数池化 | `norm_type`, `kernel_size` | 计算p-范数 |
| `nn.AvgPool1d/2d/3d` | 平均池化 | `kernel_size`, `stride` | 窗口内平均值 |

**区别**：
- 普通池化：固定窗口，可能改变输出大小
- 自适应池化：固定输出大小，自动调整窗口
- MaxPool：保留最显著特征
- AvgPool：平滑特征

### **4. 填充层**
| 模块 | 作用 | 主要参数 |
|------|------|---------|
| `nn.ReflectionPad1d/2d` | 反射填充 | `padding` |
| `nn.ReplicationPad1d/2d` | 复制填充 | `padding` |
| `nn.ZeroPad2d` | 零填充 | `padding` |
| `nn.ConstantPad1d/2d/3d` | 常数填充 | `padding`, `value` |

**使用场景**：
- 保持特征图大小
- 处理边界效应
- 调整输入输出尺寸

---

## **容器（Containers）**

### **1. 顺序容器**
| 模块 | 作用 | 特点 |
|------|------|------|
| `nn.Sequential` | 顺序执行容器 | 按定义顺序执行各层 |
| `nn.ModuleList` | 模块列表 | 存储Module的list，可迭代 |
| `nn.ModuleDict` | 模块字典 | 存储Module的dict，可按名访问 |
| `nn.ParameterList` | 参数列表 | 存储Parameter的list |
| `nn.ParameterDict` | 参数字典 | 存储Parameter的dict |

**区别**：
- `Sequential`：自动实现forward，用于简单线性结构
- `ModuleList`：需要自定义forward，用于动态或复杂结构
- `ModuleDict`：通过名字访问模块

### **2. 示例对比**
```python
# Sequential - 自动forward
model1 = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

# ModuleList - 手动forward
class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

---

## **激活函数（Activation Functions）**

| 模块 | 公式 | 特点 | 适用场景 |
|------|------|------|---------|
| `nn.ReLU` | max(0, x) | 计算简单，缓解梯度消失 | 大多数隐藏层 |
| `nn.LeakyReLU` | max(αx, x) | 解决"神经元死亡"问题 | 替代ReLU |
| `nn.PReLU` | max(αx, x) | 可学习的α参数 | 需要自适应负斜率 |
| `nn.RReLU` | 随机α | 训练时α随机，测试时固定 | 正则化效果 |
| `nn.ELU` | x if x>0 else α(exp(x)-1) | 输出均值接近0 | 替代ReLU |
| `nn.CELU` | x if x>0 else α(exp(x/α)-1) | ELU的连续可导版本 | 需要平滑激活 |
| `nn.SELU` | λx if x>0 else λα(exp(x)-1) | 自归一化网络 | SNNs |
| `nn.GELU` | xΦ(x) | Transformer常用 | BERT、GPT等 |
| `nn.Sigmoid` | 1/(1+exp(-x)) | 输出(0,1) | 二分类输出层 |
| `nn.Tanh` | tanh(x) | 输出(-1,1) | RNN隐藏层 |
| `nn.Softmax` | exp(x_i)/Σexp(x_j) | 输出概率分布 | 多分类输出层 |
| `nn.LogSoftmax` | log(Softmax(x)) | 数值稳定 | 配合NLLLoss |
| `nn.Softmin` | exp(-x_i)/Σexp(-x_j) | 反向Softmax | 特殊需求 |
| `nn.Softplus` | log(1+exp(x)) | 平滑的ReLU | 回归问题 |
| `nn.Softsign` | x/(1+\|x\|) | 平滑的Tanh | 替代Tanh |
| `nn.Hardtanh` | clamp(x, min, max) | 裁剪的线性函数 | 量化网络 |
| `nn.Hardshrink` | x if \|x\|>λ else 0 | 硬收缩 | 稀疏编码 |
| `nn.Softshrink` | sign(x)max(\|x\|-λ, 0) | 软收缩 | 去噪 |
| `nn.Tanhshrink` | x - tanh(x) | Tanh的残差 | 自编码器 |

**选择指南**：
- **默认选择**：ReLU
- **梯度问题**：LeakyReLU, ELU
- **输出层**：Sigmoid(二分类), Softmax(多分类)
- **NLP/Transformer**：GELU

---

## **损失函数（Loss Functions）**

### **1. 回归损失**
| 模块 | 公式 | 特点 | 适用场景 |
|------|------|------|---------|
| `nn.L1Loss` | \|y-ŷ\| | 绝对误差，对异常值不敏感 | 稳健回归 |
| `nn.MSELoss` | (y-ŷ)² | 均方误差，最常见 | 一般回归问题 |
| `nn.SmoothL1Loss` | 0.5(y-ŷ)²/β if \|y-ŷ\|<β else \|y-ŷ\|-0.5β | Huber损失，结合L1和MSE | 目标检测 |
| `nn.HuberLoss` | 同SmoothL1Loss | 对异常值鲁棒 | 稳健回归 |

### **2. 分类损失**
| 模块 | 公式/描述 | 特点 | 适用场景 |
|------|----------|------|---------|
| `nn.CrossEntropyLoss` | -log(exp(ŷ_class)/Σexp(ŷ)) | Softmax+负对数似然 | 多分类（默认） |
| `nn.NLLLoss` | -ŷ[class] | 负对数似然，需LogSoftmax输入 | 配合LogSoftmax |
| `nn.BCELoss` | -(y*log(ŷ)+(1-y)*log(1-ŷ)) | 二值交叉熵，输入需在(0,1) | 二分类（需Sigmoid） |
| `nn.BCEWithLogitsLoss` | BCE+Sigmoid | 包含Sigmoid的BCE | 二分类（推荐） |
| `nn.MultiLabelSoftMarginLoss` | 多标签BCE | 每个类别独立二分类 | 多标签分类 |
| `nn.MultiLabelMarginLoss` | 多类别多分类Hinge损失 | 支持多标签 | 多标签分类 |
| `nn.MultiMarginLoss` | 多分类Hinge损失 | 支持多类别 | 多分类替代方案 |
| `nn.KLDivLoss` | KL散度 | 衡量两个分布差异 | 分布匹配 |
| `nn.PoissonNLLLoss` | 泊松负对数似然 | 计数数据 | 计数回归 |
| `nn.CosineEmbeddingLoss` | 余弦相似度损失 | 衡量向量相似度 | 相似度学习 |
| `nn.MarginRankingLoss` | 排序损失 | 学习排序 | 推荐系统 |
| `nn.TripletMarginLoss` | 三元组损失 | 学习嵌入空间 | 度量学习 |
| `nn.HingeEmbeddingLoss` | Hinge嵌入损失 | 半监督学习 | 嵌入学习 |
| `nn.CTCLoss` | 时序分类损失 | 对齐序列 | 语音识别 |

### **3. 选择指南**
```python
# 二分类：BCEWithLogitsLoss
loss = nn.BCEWithLogitsLoss()

# 多分类（单标签）：CrossEntropyLoss  
loss = nn.CrossEntropyLoss()  # 最常用

# 多分类（多标签）：MultiLabelSoftMarginLoss
loss = nn.MultiLabelSoftMarginLoss()

# 回归：MSELoss 或 SmoothL1Loss
loss = nn.MSELoss()  # 一般回归
loss = nn.SmoothL1Loss()  # 目标检测
```

---

## **优化器（Optimizers）**

### **1. 基础优化器**
| 优化器 | 更新公式 | 特点 | 适用场景 |
|--------|---------|------|---------|
| `optim.SGD` | θ = θ - η∇θ | 随机梯度下降 | 所有场景 |
| `optim.SGD+momentum` | v = γv + η∇θ; θ = θ - v | 带动量的SGD | 加速收敛 |
| `optim.Adam` | 自适应矩估计 | 自适应学习率+动量 | 深度学习默认 |
| `optim.AdamW` | Adam+解耦权重衰减 | 改进的Adam | 现代架构默认 |
| `optim.Adagrad` | η/(√G+ε) | 自适应学习率 | 稀疏数据 |
| `optim.Adadelta` | 自适应学习率 | 无学习率版本 | 替代Adagrad |
| `optim.RMSprop` | EMA梯度平方 | RNN优化 | RNN/LSTM |
| `optim.Adamax` | Adam的无穷范数版 | 更稳定 | 替代Adam |
| `optim.NAdam` | Adam+Nesterov动量 | 加速收敛 | 替代Adam |
| `optim.Rprop` | 弹性反向传播 | 仅符号，无视大小 | 全批量数据 |
| `optim.LBFGS` | 拟牛顿法 | 二阶优化 | 小批量数据 |
| `optim.SparseAdam` | Adam的稀疏版 | 稀疏梯度 | 嵌入层 |

### **2. 参数对比**
```python
# SGD（最基础）
optim.SGD(params, lr=0.01, momentum=0, dampening=0, 
          weight_decay=0, nesterov=False)

# Adam（最常用）
optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08,
           weight_decay=0, amsgrad=False)

# AdamW（推荐）
optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08,
            weight_decay=0.01, amsgrad=False)
```

### **3. 选择策略**
- **新手/默认**：Adam 或 AdamW
- **需要最好结果**：SGD+momentum（需调参）
- **RNN/LSTM**：RMSprop 或 Adam
- **小数据集**：LBFGS（二阶优化）
- **稀疏特征**：Adagrad 或 SparseAdam

---

## **学习率调度器（Learning Rate Schedulers）**

### **1. 基于步数的调度器**
| 调度器 | 策略 | 参数 | 特点 |
|--------|------|------|------|
| `lr_scheduler.LambdaLR` | 自定义函数 | `lr_lambda` | 最灵活 |
| `lr_scheduler.StepLR` | 固定步长衰减 | `step_size`, `gamma` | 简单常用 |
| `lr_scheduler.MultiStepLR` | 多步长衰减 | `milestones`, `gamma` | 灵活步长 |
| `lr_scheduler.ExponentialLR` | 指数衰减 | `gamma` | 平滑衰减 |
| `lr_scheduler.CosineAnnealingLR` | 余弦退火 | `T_max`, `eta_min` | 周期性重启 |
| `lr_scheduler.CosineAnnealingWarmRestarts` | 带热重启余弦 | `T_0`, `T_mult` | 改善收敛 |
| `lr_scheduler.CyclicLR` | 循环学习率 | `base_lr`, `max_lr` | 周期性变化 |
| `lr_scheduler.OneCycleLR` | 单周期策略 | `max_lr`, `total_steps` | 快速收敛 |

### **2. 基于指标的调度器**
| 调度器 | 策略 | 参数 | 特点 |
|--------|------|------|------|
| `lr_scheduler.ReduceLROnPlateau` | 平台期衰减 | `factor`, `patience` | 自适应调整 |
| `lr_scheduler.ChainedScheduler` | 链式调度 | 调度器列表 | 组合策略 |

### **3. 使用示例**
```python
# StepLR：每30个epoch学习率减半
scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

# ReduceLROnPlateau：验证损失不改善时衰减
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# CosineAnnealingWarmRestarts：带热重启的余弦退火
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# 训练循环中使用
for epoch in range(100):
    train(...)
    val_loss = validate(...)
    
    # StepLR方式
    scheduler.step()
    
    # ReduceLROnPlateau方式  
    scheduler.step(val_loss)
```

---

## **归一化层（Normalization Layers）**

| 模块 | 归一化维度 | 适用场景 | 训练/测试差异 |
|------|-----------|---------|-------------|
| `nn.BatchNorm1d` | (N, C)或(N, C, L) | 全连接层、序列 | 使用running统计 |
| `nn.BatchNorm2d` | (N, C, H, W) | 卷积网络 | 使用running统计 |
| `nn.BatchNorm3d` | (N, C, D, H, W) | 3D卷积 | 使用running统计 |
| `nn.LayerNorm` | (N, C, *)最后维度 | NLP、Transformer | 无running统计 |
| `nn.InstanceNorm1d/2d/3d` | (N, C, *)逐样本 | 风格迁移、GAN | 无running统计 |
| `nn.GroupNorm` | (N, C, *)分组 | 小batch训练 | 无running统计 |
| `nn.LocalResponseNorm` | 局部响应归一化 | AlexNet | 通道间归一化 |
| `nn.SyncBatchNorm` | 同步BatchNorm | 多GPU训练 | 跨GPU同步 |

**区别**：
- **BatchNorm**：批次维度归一化，batch_size小时效果差
- **LayerNorm**：样本维度归一化，适合变长序列
- **InstanceNorm**：样本+通道维度，适合风格化
- **GroupNorm**：分组归一化，batch_size小时替代BatchNorm

**选择指南**：
- **CNN图像**：BatchNorm2d
- **NLP/Transformer**：LayerNorm
- **风格迁移**：InstanceNorm2d
- **小batch训练**：GroupNorm

---

## **循环神经网络层（RNN Layers）**

### **1. 基础RNN层**
| 模块 | 结构 | 特点 | 参数 |
|------|------|------|------|
| `nn.RNN` | 基本RNN | 简单，梯度易消失 | `input_size`, `hidden_size`, `num_layers` |
| `nn.LSTM` | 长短时记忆 | 解决长依赖，最常用 | 同RNN，加`cell`状态 |
| `nn.GRU` | 门控循环单元 | LSTM简化版，计算快 | 同RNN，少一个门 |
| `nn.RNNCell` | 单步RNN单元 | 手动控制循环 | 单时间步 |
| `nn.LSTMCell` | 单步LSTM单元 | 手动控制循环 | 单时间步 |
| `nn.GRUCell` | 单步GRU单元 | 手动控制循环 | 单时间步 |

### **2. 参数详解**
```python
# 创建LSTM层
lstm = nn.LSTM(
    input_size=100,      # 输入特征维度
    hidden_size=256,     # 隐藏状态维度
    num_layers=2,        # 堆叠层数
    batch_first=True,    # 输入形状(batch, seq, feature)
    dropout=0.5,         # 层间dropout（最后一层无）
    bidirectional=True   # 双向LSTM
)

# 输入输出形状
# 输入: (batch, seq_len, input_size)
# 输出: (batch, seq_len, hidden_size * num_directions)
# 隐藏状态: (num_layers * num_directions, batch, hidden_size)
```

### **3. 选择指南**
- **需要捕捉长依赖**：LSTM（默认）
- **计算资源有限**：GRU（更快更省内存）
- **简单序列模式**：RNN
- **需要精细控制**：RNNCell/LSTMCell/GRUCell

---

## **Transformer层（Transformer Layers）**

### **1. 核心组件**
| 模块 | 作用 | 参数 | 特点 |
|------|------|------|------|
| `nn.Transformer` | 完整Transformer | `d_model`, `nhead`, `num_layers` | 编码器-解码器 |
| `nn.TransformerEncoder` | Transformer编码器 | 同Transformer | 仅编码器 |
| `nn.TransformerDecoder` | Transformer解码器 | 同Transformer | 仅解码器 |
| `nn.TransformerEncoderLayer` | 编码器层 | `d_model`, `nhead`, `dim_feedforward` | 单层编码器 |
| `nn.TransformerDecoderLayer` | 解码器层 | 同编码器层 | 单层解码器 |
| `nn.MultiheadAttention` | 多头注意力 | `embed_dim`, `num_heads` | 可单独使用 |

### **2. 使用示例**
```python
# 创建Transformer编码器
encoder_layer = nn.TransformerEncoderLayer(
    d_model=512,           # 嵌入维度
    nhead=8,               # 注意力头数
    dim_feedforward=2048,  # 前馈网络维度
    dropout=0.1
)

transformer_encoder = nn.TransformerEncoder(
    encoder_layer, 
    num_layers=6          # 编码器层数
)

# 创建多头注意力
attention = nn.MultiheadAttention(
    embed_dim=512,
    num_heads=8,
    dropout=0.1,
    batch_first=True      # 输入形状(batch, seq, feature)
)
```

### **3. 适用场景**
- **机器翻译**：完整Transformer
- **文本分类**：TransformerEncoder
- **生成任务**：TransformerDecoder
- **预训练模型**：BERT、GPT等基于Transformer

---

## **实用工具（Utility Modules）**

### **1. Dropout层**
| 模块 | 作用 | 参数 | 训练/测试差异 |
|------|------|------|-------------|
| `nn.Dropout` | 随机失活 | `p`（失活概率） | 训练时失活，测试时缩放 |
| `nn.Dropout2d` | 2D Dropout | `p` | 通道维度失活 |
| `nn.Dropout3d` | 3D Dropout | `p` | 通道维度失活 |
| `nn.AlphaDropout` | Alpha Dropout | `p` | 保持自归一化属性 |
| `nn.FeatureAlphaDropout` | 特征Alpha Dropout | `p` | 通道维度Alpha Dropout |

**区别**：
- `Dropout`：元素级失活
- `Dropout2d/3d`：通道级失活（对CNN更有效）
- `AlphaDropout`：保持均值和方差，适合SELU激活

### **2. 嵌入层**
| 模块 | 作用 | 参数 | 输出形状 |
|------|------|------|---------|
| `nn.Embedding` | 词嵌入 | `num_embeddings`, `embedding_dim` | (*) → (*, embedding_dim) |
| `nn.EmbeddingBag` | 嵌入包 | 同Embedding | 支持聚合模式 |

**区别**：
- `Embedding`：将索引映射为向量
- `EmbeddingBag`：高效计算嵌入的均值/和等

### **3. 距离函数**
| 模块 | 计算 | 应用 |
|------|------|------|
| `nn.PairwiseDistance` | 向量对距离 | 相似度计算 |
| `nn.CosineSimilarity` | 余弦相似度 | 相似度计算 |

### **4. 其他工具**
| 模块 | 作用 | 示例 |
|------|------|------|
| `nn.Flatten` | 展平输入 | 卷积→全连接过渡 |
| `nn.Unflatten` | 反展平 | 恢复形状 |
| `nn.Fold` | 滑动窗口组合 | 重叠块重组 |
| `nn.Unfold` | 滑动窗口提取 | 提取局部块 |
| `nn.ChannelShuffle` | 通道重排 | ShuffleNet |
| `nn.Upsample` | 上采样 | 放大特征图 |
| `nn.PixelShuffle` | 像素重排 | 亚像素卷积 |
| `nn.ZeroPad2d` | 零填充 | 调整边界 |
| `nn.ConstantPad2d` | 常数填充 | 调整边界 |

---

## **功能接口（Functional Interface）**

### **1. 什么是Functional？**
```python
import torch.nn.functional as F

# 与nn.Module的区别
# nn.Module版本（有状态，可学习参数）
relu_layer = nn.ReLU()
output = relu_layer(input)

# Functional版本（无状态，纯函数）
output = F.relu(input)
```

### **2. 常用Functional函数**
| 类别 | 函数示例 | 对应Module |
|------|---------|-----------|
| 激活函数 | `F.relu()`, `F.sigmoid()`, `F.softmax()` | `nn.ReLU`, `nn.Sigmoid`, `nn.Softmax` |
| 损失函数 | `F.cross_entropy()`, `F.mse_loss()` | `nn.CrossEntropyLoss`, `nn.MSELoss` |
| 卷积操作 | `F.conv2d()`, `F.max_pool2d()` | `nn.Conv2d`, `nn.MaxPool2d` |
| 线性操作 | `F.linear()` | `nn.Linear` |
| Dropout | `F.dropout()`, `F.dropout2d()` | `nn.Dropout`, `nn.Dropout2d` |
| 归一化 | `F.batch_norm()`, `F.layer_norm()` | `nn.BatchNorm2d`, `nn.LayerNorm` |
| 嵌入 | `F.embedding()` | `nn.Embedding` |
| 注意力 | `F.scaled_dot_product_attention()` | - |
| 距离 | `F.pairwise_distance()`, `F.cosine_similarity()` | `nn.PairwiseDistance`, `nn.CosineSimilarity` |

### **3. 使用场景**
- **需要灵活性时**：在自定义forward中使用Functional
- **无参数操作**：激活函数、池化等
- **快速原型**：简单测试时
- **自定义层**：实现特殊操作时

```python
class CustomLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10, 10))
    
    def forward(self, x):
        # 使用functional进行计算
        x = F.linear(x, self.weight)  # 自定义线性变换
        x = F.relu(x)                 # 激活函数
        x = F.dropout(x, p=0.5, training=self.training)  # Dropout
        return x
```

---

## **模块选择快速参考表**

| 任务类型 | 推荐层 | 推荐激活 | 推荐损失 | 推荐优化器 |
|---------|--------|---------|---------|-----------|
| **图像分类** | Conv2d, BatchNorm2d | ReLU | CrossEntropyLoss | AdamW |
| **目标检测** | Conv2d, BatchNorm2d | ReLU | SmoothL1Loss, CrossEntropyLoss | SGD+momentum |
| **语义分割** | Conv2d, BatchNorm2d | ReLU | CrossEntropyLoss, DiceLoss | AdamW |
| **语音识别** | Conv1d, LSTM, BatchNorm1d | ReLU | CTCLoss | Adam |
| **机器翻译** | Transformer, Embedding | GELU | CrossEntropyLoss | AdamW |
| **文本分类** | Embedding, LSTM, Linear | ReLU/Tanh | CrossEntropyLoss | Adam |
| **生成对抗网络** | ConvTranspose2d, BatchNorm2d | LeakyReLU | BCEWithLogitsLoss | Adam |
| **推荐系统** | Embedding, Linear | ReLU | BCEWithLogitsLoss | Adam |
| **时间序列预测** | LSTM, Linear | ReLU | MSELoss | Adam |
| **函数拟合** | Linear | ReLU | MSELoss | Adam |

---

## **总结**

PyTorch的模块系统设计得非常全面和灵活：

1. **层次清晰**：从底层操作到高层架构都有对应模块
2. **灵活组合**：容器类允许任意复杂的网络结构
3. **覆盖全面**：涵盖传统CNN/RNN到现代Transformer
4. **易于扩展**：可以轻松实现自定义模块

**核心原则**：
- 使用`nn.Module`管理可学习参数和复杂结构
- 使用`nn.functional`进行无状态操作
- 根据任务选择合适模块组合
- 通过容器组织复杂网络结构

这个模块系统使得PyTorch既能用于学术研究（灵活），又能用于工业生产（高效）。