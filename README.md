# 乳腺癌良恶性分类 — 五模型综合对比

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

本项目是机器学习课程大作业的完整实现。五位成员分别采用 K 近邻、支持向量机、随机森林、梯度提升、多层感知机五种算法，对威斯康星乳腺癌数据集进行良恶性二分类。每位成员在自己的子报告中独立完成数据探索、模型训练和 GridSearchCV 超参数调优。最终在统一的 80/20 数据划分和 StandardScaler 标准化条件下，将各成员的最优模型配置进行公平对比，生成四维度对比可视化图表和深度分析报告。

## 数据集

**Breast Cancer Wisconsin (Diagnostic) Data Set**

- 来源：[Kaggle / UCI ML Repository](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- 样本数：569 例（恶性 212 例，良性 357 例）
- 特征数：30 个数值特征（细胞核均值/标准差/最差值各 10 个）
- 任务：二分类（恶性 Malignant / 良性 Benign）

## 组员与分工

| 姓名 | 学号 | 算法 | 最优配置（GridSearchCV） | 子报告 |
|------|------|------|-------------------------|--------|
| 巨峰 | 2023217199 | 梯度提升 (GB) | `n=100, lr=0.1, max_depth=3` | `GradientBoosting/` |
| 韩明扬 | 2023217191 | 支持向量机 (SVM) | `kernel=linear, C=100` | `机器学习/` |
| 刘志富 | 2023216547 | 随机森林 (RF) | `n=100, max_depth=10, min_split=5, min_leaf=2` | `dazuoye/` |
| 江玮 | 2023217211 | K 近邻 (KNN) | `k=4, uniform, L1距离` | `KNN-jw/` |
| 周子烨 | 2023217220 | 多层感知机 (MLP) | `hidden=(100,50), relu, early_stopping` | `MLP/` |

## 实验结果

统一数据划分（80/20 分层抽样，StandardScaler，random_state=42），各成员最优配置：

| 算法 | 准确率 | 精确率 | 召回率 | F1 | 5折CV±σ |
|------|--------|--------|--------|-----|----------|
| 逻辑回归 (Baseline) | **0.9825** | 0.9861 | 0.9861 | **0.9861** | 0.9780±0.0098 |
| SVM (韩明扬) | 0.9737 | 0.9726 | 0.9861 | 0.9793 | 0.9429±0.0146 |
| KNN (江玮) | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9670±0.0139 |
| 随机森林 (刘志富) | 0.9561 | **0.9589** | 0.9722 | 0.9655 | 0.9626±0.0179 |
| 梯度提升 (巨峰) | 0.9561 | 0.9467 | **0.9861** | 0.9660 | 0.9516±0.0149 |
| MLP (周子烨) | 0.9386 | 0.9452 | 0.9583 | 0.9517 | 0.9187±0.0420 |

## 关键发现

1. **逻辑回归最优**（准确率 98.25%）：数据集线性可分性极强，简单模型的泛化能力超越复杂模型
2. **梯度提升召回率最高**（98.61%，仅漏诊 1 例）：适合乳腺癌早期筛查，Low FN 是第一优先级
3. **KNN 以极简配置持平复杂模型**：k=4 + 曼哈顿距离即可达到 95.61%，证明数据质量 > 模型复杂度
4. **MLP CV 波动最大**（±0.0420）：深度模型在 569 样本上过参数化，early_stopping 不能完全解决
5. **全部模型 AUC > 0.98**：数据集质量高，各种方法均能有效区分良恶性

详见 `乳腺癌分类_五模型综合对比.ipynb` 第 8 节深度分析。

## 项目结构

```
├── 乳腺癌分类_五模型综合对比.ipynb   # 综合对比 Notebook（含深度分析）
├── README.md
├── figures/                         # 对比可视化图表
│   ├── model_comparison_metrics.png
│   ├── cv_boxplot.png
│   ├── roc_curves_comparison.png
│   └── confusion_matrices_comparison.png
├── GradientBoosting/                # 巨峰 · 梯度提升子报告
├── 机器学习/                         # 韩明扬 · SVM 子报告
├── dazuoye/                         # 刘志富 · 随机森林子报告
├── KNN-jw/                          # 江玮 · KNN 子报告
└── MLP/                             # 周子烨 · MLP 子报告
```

## 快速开始

### 环境

- Python >= 3.8
- Jupyter Notebook / JupyterLab

### 安装

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 运行

```bash
# 综合对比（推荐）
jupyter notebook 乳腺癌分类_五模型综合对比.ipynb

# 各成员子报告
jupyter notebook GradientBoosting/GradientBoosting_乳腺癌分类.ipynb
jupyter notebook 机器学习/Breast_Cancer_SVM.ipynb
jupyter notebook dazuoye/Breast_Cancer_RF_Classification随机森林.ipynb
jupyter notebook KNN-jw/KNN_乳腺癌分类.ipynb
jupyter notebook MLP/MLP.ipynb
```

选择 Run All 即可从头执行完整分析并生成所有图表。

> 数据集通过 `sklearn.datasets.load_breast_cancer()` 直接加载，无需手动下载。Windows 用户需确保系统安装了 SimHei 字体以正常显示图表中文标题。

## 依赖

- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/) + SimHei 中文字体
- [Seaborn](https://seaborn.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)

## 许可证

本项目仅用于学术研究与学习目的。

---

**课程**：机器学习 · 大作业  
**数据集**：Breast Cancer Wisconsin (Diagnostic) Data Set  
**GitHub**：[Bonjour-1](https://github.com/Bonjour-1)
