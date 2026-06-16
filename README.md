# Machine Learning Course Project: Semi-supervised MNIST Classification

本仓库是《机器学习》课程大作业的代码仓库，主题是在 MNIST 上完成极少标签条件下的半监督手写数字分类。

作业约束很严格：划分后的 `train_dataset` 中只有题目指定的 10 张图像允许使用真实标签，每个数字类别各 1 张；其余训练图像必须作为无标签数据处理，不能用于训练或超参数选择。最终目标是在不违反标签约束的前提下，训练一个在 `mnist_test` 上表现较好的 CNN 分类器。

最终方案为：

```text
SimCLR representation -> LGC label propagation -> CNN classifier -> self-training
```

最佳测试集准确率为 **98.55%**。

## Project Scope

本仓库包含：

- 数据划分与 10 个锚点标签校验代码。
- 三种表征路线：原始像素、卷积自编码机、SimCLR。
- 两种伪标签方法：LGC 标签传播和最近原型基线。
- CNN 分类器训练与自训练流程。
- 对比实验、传播方法对比、消融实验和可视化脚本。
- 报告中使用的 6 张主要图像。

本仓库不包含：

- MNIST 原始数据文件。
- 已训练模型权重。
- 运行生成的 `result/` 实验输出。

这些文件体积较大，放在补充材料压缩包中。

## Supplementary Materials

补充材料文件：`大作业补充材料.zip`

百度网盘链接：

```text
https://pan.baidu.com/s/1YlzcaDg6IiaGkBl2IcaX_Q?pwd=sjxv
```

提取码：

```text
sjxv
```

补充材料包含 `data/`、`model/` 和 `result/`。将它们复制到仓库根目录后，可以直接读取已有数据、模型权重和实验结果；也可以只使用本仓库源码重新生成这些文件。

## Repository Structure

```text
core/
  config.py              全局配置、锚点索引、超参数和路径
  data.py                MNIST 加载、固定划分、锚点校验、loader 构造
  models.py              CNN backbone、Autoencoder、SimCLR、CNNClassifier
  utils.py               随机种子、设备、评估、checkpoint 工具

train/
  train_autoencoder.py   自编码机训练与 embedding 抽取
  train_contrastive.py   SimCLR 训练与 NT-Xent 损失
  train_classifier.py    CNN 训练、自训练和高置信样本扩充

propagation/
  propagate.py           LGC 标签传播、最近原型基线、置信度筛选

pipeline/
  run.py                 单条路线端到端运行
  compare.py             pixel / AE / SimCLR 三路线对比
  compare_prop.py        LGC 与最近原型传播方法对比
  tune.py                验证集超参数扫描
  ablation.py            仅传播 vs. 加自训练的消融实验

visualization/
  component_figs.py      报告方法图生成
  plot_results.py        实验结果汇总图
  tsne_viz.py            t-SNE 可视化

doc/
  1.png ... 6.png        报告主要图像
  requirement.md         作业要求整理
```

## Method Summary

整体流程分为四步：

1. **无标签表征学习**：使用全部 50,000 张训练图像，但不使用标签。比较原始像素、卷积自编码机和 SimCLR 三种表征。
2. **标签传播**：以 10 个有标签锚点为种子，在 embedding 空间构建 kNN 图，用 LGC 扩散标签，得到伪标签和置信度。
3. **CNN 分类器训练**：用 10 个真标签和高置信伪标签训练最终交付的 CNN 分类器。
4. **自训练**：用上一轮 CNN 重新给无标签样本打分，筛选高置信样本加入训练集并重训。

核心原则是：训练集中的非锚点真实标签不参与训练或超参数选择；验证集标签只用于模型选择和调参；测试集只用于最终汇报。

## Environment

实测环境：

- Python 3.13
- PyTorch 2.11
- CUDA 13.0
- NVIDIA GeForce RTX 5060 Laptop GPU

安装依赖：

```bash
pip install -r requirements.txt
```

如果需要 GPU 版本 PyTorch，请根据本机 CUDA 版本按 PyTorch 官网命令安装。

## Reproduction

所有命令均从仓库根目录运行。

检查数据划分和 10 个锚点：

```bash
python -m core.data
```

快速冒烟测试：

```bash
python -m pipeline.run --route pixel --quick
python -m pipeline.compare --quick
```

运行单条路线：

```bash
python -m pipeline.run --route pixel
python -m pipeline.run --route ae
python -m pipeline.run --route simclr
```

运行主对比实验：

```bash
python -m pipeline.compare
```

运行传播方法对比和消融实验：

```bash
python -m pipeline.compare_prop
python -m pipeline.ablation
```

重新生成报告图：

```bash
python -m visualization.method_figs
python -m visualization.plot_results
python -m visualization.tsne_viz
```

## Main Results

| Representation | Propagation coverage | Pseudo-label accuracy | Best round | Val | Test |
|---|---:|---:|---:|---:|---:|
| Pixel | 26.59% | 99.62% | 2 | 94.30% | 94.67% |
| Autoencoder | 16.28% | 99.52% | 2 | 94.70% | 95.22% |
| SimCLR | 85.87% | 99.50% | 2 | 98.42% | 98.55% |

主要结论：

- SimCLR 表征显著提高同类样本在 embedding 空间中的局部一致性。
- LGC 标签传播优于只依赖单个锚点距离的最近原型基线。
- 自训练能继续补充高置信样本，但最终性能主要由表征质量和传播覆盖率决定。

## Notes

- `data/`、`model/`、`result/` 不纳入 GitHub 版本控制，请从补充材料获取或重新运行脚本生成。
- 重新训练会覆盖 `model/encoder_*.pt` 和 `model/classifier_*.pt`。
- `--quick` 只用于流程测试，不应作为最终准确率汇报。
- 伪标签准确率和 t-SNE 真实标签着色只用于诊断分析，不回流到训练或调参。
