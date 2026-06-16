# MNIST 极少标签半监督分类（每类 1 个标签）

机器学习大作业。`train_dataset`（5 万张）里**只有 10 张图允许使用标签**（每类 1 张，索引由作业给定），
其余约 4.999 万张为无标签数据。目标：充分利用无标签数据，训练出在 `mnist_test` 上准确率尽量高的 **CNN**。

## 方法总览（两条表征路线 + 图标签传播 + 自训练）

```
                       ┌──────────────────────────────────────────┐
  全部 5 万张训练图 ──▶ │ Stage1 无监督表征（二选一/对比）            │
   (无标签)            │   A. 卷积自编码机 (重建)                    │
                       │   B. SimCLR 对比学习 (NT-Xent, 无翻转增广)  │
                       └───────────────────┬──────────────────────┘
                                           │ 嵌入 emb (N,128)
              10 个锚点标签 ───────────────▼──────────────────────┐
                       │ Stage2 标签传播 (Zhou 2004, kNN 图)        │ → 伪标签 + 置信度
                       └───────────────────┬──────────────────────┘
                                           ▼
                       │ Stage3 训 CNN：10 真标签 + 高置信伪标签     │
                       │ Stage4 自训练：CNN 重新打分→扩样本→重训     │ ← 超参只用 val 选
                       └───────────────────┬──────────────────────┘
                                           ▼  在 mnist_test 上汇报
```

## 文件结构（已按功能归档为子包，从仓库根用 `python -m` 运行）

| 文件 | 作用 |
|---|---|
| `core/config.py` | 中央配置：种子、锚点索引、各阶段超参、路径 |
| `core/utils.py` | 种子、设备、评估、检查点 |
| `core/data.py` | **逐字复现作业划分**；锚点校验；有/无标签切分；各类 loader |
| `core/models.py` | 共享 backbone、自编码机、SimCLR、最终 `CNNClassifier` |
| `train/train_autoencoder.py` | 路线 A：卷积自编码机 + 抽嵌入 |
| `train/train_contrastive.py` | 路线 B：SimCLR 自监督 + 抽嵌入 |
| `propagation/propagate.py` | 标签传播（主）+ 最近原型（基线）+ 置信度筛选 |
| `train/train_classifier.py` | CNN 训练 + 自训练迭代（按 val 选最优） |
| `pipeline/run.py` | 端到端跑单条路线 |
| `pipeline/compare.py` | 像素/AE/SimCLR 三路线对比，输出 `result/comparison.csv` |
| `pipeline/tune.py` | val 上扫超参（knn_k × tau_prop） |
| `pipeline/compare_prop.py` | 传播方法对比：LP vs 最近原型基线 |
| `pipeline/ablation.py` | 消融表（仅传播 vs +自训练） |
| `visualization/tsne_viz.py` | 三嵌入 t-SNE 可视化 |
| `visualization/plot_results.py` | 结果汇总多面板图 |

## 运行

```bash
pip install -r requirements.txt        # 见文件内 GPU torch 安装说明

# 均从仓库根目录用 python -m 运行（包内绝对导入）；首次会下载 MNIST 到 ./data、缓存嵌入到 ./result
python -m pipeline.run --route ae
python -m pipeline.run --route simclr

# 三路线对比（推荐，报告用）
python -m pipeline.compare

# 快速冒烟（极小 epoch，仅验证流程打通，非真实精度）
python -m pipeline.run --route pixel --quick
python -m pipeline.compare --quick
```

常用开关：`--tau_prop` 传播阶段阈值(默认 0.7)、`--tau_self` 自训练阶段阈值(默认 0.95)、
`--knn_k` 近邻数、`--self_train_rounds` 自训练轮数、`--propagation {lp,proto}`、
`--no-cache`（改了嵌入超参后强制重训）。

> 注意两个阈值刻度不同：标签传播概率较"软"(经验 0.5–0.9，取 ~0.7)，CNN softmax 过度自信(取 ~0.95)。

## 合规要点（务必遵守作业约束）

- ✅ 只用 `config.ANCHOR_INDICES` 这 10 张的真标签训练；`data.verify_anchors` 会断言它们恰为每类一张。
- ❌ `train_dataset` 其余图片的真标签**不得用于训练或选超参**。
- ✅ `val_dataset` 标签**允许**用于选超参（`tau_prop`、`tau_self`、轮数、epoch 等都按 val 准确率选）。
- ⚠️ 代码里统计“伪标签准确率”等指标仅为**报告诊断**，由 `config.diagnostic_use_true_labels`
  控制，**绝不回流**到训练或模型选择；提交“纯净版”可将其设为 `False`。

## 产物（`./result/`）

`emb_{route}.npy`（缓存嵌入）、`encoder_{route}.pt`、`classifier_{route}.pt`、
`result_{route}.json`（逐轮 val/test）、`comparison.csv`、`comparison_propagation.csv`、
`tune_simclr.csv`、`ablation.csv`、`tsne_routes.png`、`results_summary.png`。报告在 `doc/`。

## 复现性

种子固定 42；`utils.seed_torch` 复用作业给定设置。划分仅由 `torch.Generator(seed=42)` 决定，
与下载目录无关。本机已验证 10 个锚点 = 0–9 每类一张，划分复现正确。
