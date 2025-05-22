# QGRL: Quaternion Graph Representation Learning for Heterogeneous Feature 

Data Clustering

QGRL 是一个基于四元数图表示学习的异构特征数据聚类方法。该方法通过四元数神经网络来学习数据的低
维表示，并利用图结构来捕获数据样本之间的关系，从而实现高效的聚类。

## 主要特点

-  基于四元数神经网络的特征学习
-  图结构感知的表示学习
-  端到端的无监督聚类框架
-  支持多种数据集的异构特征处理

## 环境要求

- Python 3.6+
- PyTorch >= 1.6.0
- scikit-learn
- numpy
- scipy
- munkres

## 项目结构

```

QGRL/
├── core_qnn/ # 四元数神经网络核心实现
│ ├── quaternion_layers.py # 四元数神经网络层定义
│ └── quaternion_ops.py # 四元数基础运算操作
├── MLdata/ # 数据集目录
├── metrics.py # 评估指标实现
├── model.py # QGRL 模型定义
├── run.py # 训练和评估脚本
└── Quaternion_MLdata_load.py # 数据加载工具

```

##  快速开始

1.  安装依赖：

```bash
pip install -r requirements.txt
```

2. 运行实验：

```
python run.py
```

## 参数配置

主要的超参数包括：

- layers : 网络层维度配置，默认[512, 256, 128]
- learning_rate : 学习率，默认 4e-4
- max_epoch : 最大训练轮数，默认 50
- max_iter : 每轮迭代次数，默认 1
- pre_iter : 预训练迭代次数，默认 10
- coeff_reg : 正则化系数，默认 0.0001

## 支持的数据集

- zoo (7 类)
- iris (3 类)
- wine (3 类)
- car (4 类)
- heart (2 类)
- ttt (2 类)
- yeast (10 类)
- breast (2 类)
- hayes (3 类)
- glass (6 类)

## 主要模块说明

- QGRL : 主模型类，实现了四元数图表示学习的核心算法
- QGNNLayer : 四元数图神经网络层的实现
- metrics.py : 包含聚类评估指标的实现，如 ACC、NMI、ARI 等
- quaternion_ops.py : 实现了四元数相关的基础运算操作

## 引用

如果您在研究中使用了 QGRL，请引用我们的论文：

```
@inproceedings{
QGRL,
title={{QGRL}: Quaternion Graph Representation Learning for Heterogeneous 
Feature Data Clustering},
author={Junyang Chen, Yuzhu Ji, Rong ZOU, Yiqun Zhang, Yiu-ming Cheung},
booktitle={Proceedings of the 30th SIGKDD Conference on Knowledge Discovery 
and Data Mining},
year={2024}
}
```

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 联系方式

如有任何问题，请通过以下方式联系我们：

- 提交 Issue
- 发送邮件至[项目维护者邮箱]
